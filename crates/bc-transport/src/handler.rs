//! The [`FlightService`] implementation backing a `FlightServer`, plus the
//! credit-grant encode/decode helpers it shares with the exchange client.

use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::FlightService;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::stream::{BoxStream, StreamExt, TryStreamExt};
use tokio::sync::Semaphore;
use tonic::{Request, Response, Status, Streaming};

use crate::store::PartitionStore;

/// The [`FlightService`] implementation backing a [`FlightServer`].
///
/// [`FlightServer`]: crate::FlightServer
#[derive(Clone)]
pub(crate) struct FlightHandler {
    pub(crate) store: Arc<PartitionStore>,
    /// Optional shared-secret token. When set, a `do_exchange` whose first message
    /// does not carry a matching token (in `flight_descriptor.path[1]`) is rejected
    /// with `Unauthenticated` — so a process that can merely reach the port cannot
    /// exfiltrate shuffle partitions (N5). `None` disables the check (single-host /
    /// trusted-network default).
    pub(crate) token: Option<String>,
}

type DoGetStream = BoxStream<'static, Result<FlightData, Status>>;

#[tonic::async_trait]
impl FlightService for FlightHandler {
    type HandshakeStream = BoxStream<'static, Result<HandshakeResponse, Status>>;
    type ListFlightsStream = BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = DoGetStream;
    type DoPutStream = BoxStream<'static, Result<PutResult, Status>>;
    type DoExchangeStream = BoxStream<'static, Result<FlightData, Status>>;
    type DoActionStream = BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = BoxStream<'static, Result<ActionType, Status>>;

    // NOTE (C45): `do_get` is the *un-credited* fetch — it encodes a whole partition
    // with no flow control, so a slow consumer could accumulate unbounded in-flight
    // frames. It is NOT on the production reducer path (which is the credit-bounded
    // `do_exchange`) and is not reachable from `bc-py`; it is retained only for the
    // crate's basic publish/fetch roundtrip tests. Do not wire it into the engine.
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = String::from_utf8(request.into_inner().ticket.to_vec())
            .map_err(|e| Status::invalid_argument(format!("ticket not valid utf-8: {e}")))?;

        let batches = self
            .store
            .get(&ticket)
            .await
            .ok_or_else(|| Status::not_found(format!("unknown ticket: {ticket}")))?;

        // Empty partition: still send the schema so the reducer can reconstruct
        // an empty-but-typed result. Pick a schema from the batches if any.
        let schema = batches
            .first()
            .map(|b| b.schema())
            .unwrap_or_else(|| Arc::new(Schema::empty()));

        let batch_vec = (*batches).clone();
        let input = futures::stream::iter(batch_vec.into_iter().map(Ok));

        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(input)
            .map_err(|e| Status::internal(format!("flight encode error: {e}")));

        Ok(Response::new(stream.boxed()))
    }

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("handshake not implemented"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights not implemented"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info not implemented"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema not implemented"))
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put not implemented"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action not implemented"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions not implemented"))
    }

    /// Credit-gated producer side of the shuffle exchange.
    ///
    /// The consumer's request stream carries *credit-grant* control messages:
    /// the ticket lives in `flight_descriptor.path[0]` of the first message and
    /// each message's `app_metadata` holds a little-endian `u32` credit count.
    /// We feed those credits into a [`Semaphore`] and acquire one permit before
    /// encoding/sending each batch, so the producer never gets more than
    /// `credits` batches ahead of the consumer. See the crate-level docs.
    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        let mut inbound = request.into_inner();

        // The first message names the ticket (in flight_descriptor.path) and
        // may also carry an initial credit grant in app_metadata.
        let first = inbound
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("do_exchange: empty request stream"))?
            .map_err(|e| Status::internal(format!("do_exchange: recv error: {e}")))?;

        let ticket = first
            .flight_descriptor
            .as_ref()
            .and_then(|d| d.path.first())
            .cloned()
            .ok_or_else(|| {
                Status::invalid_argument(
                    "do_exchange: first message must carry the ticket in flight_descriptor.path",
                )
            })?;

        // Auth (N5): when a token is configured, the consumer must present a matching
        // one in path[1]. A constant-time-ish compare avoids leaking length via early
        // exit on the (tiny) token; mismatch is rejected before any data is served.
        if let Some(expected) = &self.token {
            let provided = first
                .flight_descriptor
                .as_ref()
                .and_then(|d| d.path.get(1))
                .map(String::as_str)
                .unwrap_or("");
            if provided.len() != expected.len()
                || provided
                    .bytes()
                    .zip(expected.bytes())
                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                    != 0
            {
                return Err(Status::unauthenticated("shuffle token mismatch"));
            }
        }

        let (batches, gauge) = self
            .store
            .get_with_gauge(&ticket)
            .await
            .ok_or_else(|| Status::not_found(format!("unknown ticket: {ticket}")))?;

        // Credits available to the producer. The consumer feeds this by sending
        // grant messages; we start it empty and add the first message's grant. A
        // missing/malformed seed decodes to 0, which would stall the producer
        // forever waiting for a grant that may never come; default to a safe
        // minimum window so the exchange always makes progress (C27).
        let credits = Arc::new(Semaphore::new(0));
        let decoded = decode_credits(&first.app_metadata);
        // 0 means a missing/malformed seed (a well-formed window is always >= 1);
        // fall back to a safe default rather than stalling, but never override a
        // legitimate explicit window — that would break the credit bound (C27).
        let initial = if decoded == 0 {
            crate::DEFAULT_CREDITS
        } else {
            decoded
        };
        credits.add_permits(initial as usize);

        // Pump remaining inbound grant messages into the semaphore. Lives for the
        // duration of the response stream; aborted when the stream is dropped.
        // Each post-seed grant is also a consumer *ack* (it tops up after a
        // consumed batch), so we drop the in-flight gauge per granted credit.
        let credits_for_pump = credits.clone();
        let gauge_for_pump = gauge.clone();
        let pump = tokio::spawn(async move {
            while let Some(msg) = inbound.next().await {
                match msg {
                    Ok(data) => {
                        let granted = decode_credits(&data.app_metadata);
                        if granted > 0 {
                            for _ in 0..granted {
                                gauge_for_pump.on_ack();
                            }
                            credits_for_pump.add_permits(granted as usize);
                        }
                    }
                    // Consumer hung up; stop pumping. The producer will block on
                    // the next acquire if it still needs credits, and the stream
                    // is torn down by tonic.
                    Err(_) => break,
                }
            }
        });

        let schema = batches
            .first()
            .map(|b| b.schema())
            .unwrap_or_else(|| Arc::new(Schema::empty()));
        let batch_vec = (*batches).clone();

        // Build a credit-gated source stream: await one credit per batch before
        // letting it flow into the Flight encoder. Acquiring *before* yielding is
        // what bounds the producer to `credits` batches in flight.
        let gated = async_stream::stream! {
            let _pump = pump; // keep the credit pump alive for the stream's life
            for batch in batch_vec {
                // Block here at zero credits until the consumer grants more.
                match credits.acquire().await {
                    Ok(permit) => permit.forget(),
                    // Semaphore closed (shouldn't happen) -> stop the stream.
                    Err(_) => break,
                }
                // Account this batch as in-flight the instant we hand it to the
                // encoder; the matching on_ack() fires when the consumer's
                // top-up grant arrives. The gauge's high-water mark therefore
                // bounds how far ahead of the consumer the producer ran.
                gauge.on_send();
                yield Ok(batch);
            }
        };

        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(gated)
            .map_err(|e| Status::internal(format!("flight encode error: {e}")));

        Ok(Response::new(stream.boxed()))
    }
}

/// Decode a little-endian `u32` credit grant from a control message's
/// `app_metadata`. Anything shorter than 4 bytes is treated as zero credits.
pub(crate) fn decode_credits(meta: &[u8]) -> u32 {
    if meta.len() >= 4 {
        u32::from_le_bytes([meta[0], meta[1], meta[2], meta[3]])
    } else {
        0
    }
}

/// Encode a `u32` credit grant as little-endian bytes for `app_metadata`.
pub(crate) fn encode_credits(n: u32) -> Vec<u8> {
    n.to_le_bytes().to_vec()
}
