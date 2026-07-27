//! TLS configuration for the inter-node Flight shuffle.
//!
//! The shuffle moves query data — including columns a governance policy has already
//! masked or decrypted — directly between worker processes. On any network the operator
//! does not fully control, that traffic must be encrypted and the peers mutually
//! authenticated; the shared bearer token (`handler`) stops a stray process on a trusted
//! network but not a wire sniffer or a spoofed peer.
//!
//! These types are the engine-scoped surface: the operator supplies PEM material minted
//! by *their* PKI (an internal CA, cert-manager, a cloud private-CA) and Batcher speaks
//! it. Batcher issues no certificates and runs no CA — that is a platform concern.
//!
//! * [`TlsServerConfig`] — this node's server identity, plus (for mTLS) the CA that a
//!   connecting client's certificate must chain to.
//! * [`TlsClientConfig`] — the CA the peer's server certificate must chain to, the name
//!   to verify against it, and (for mTLS) this node's client identity.

use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

use crate::{TransportError, TransportResult};

/// A PEM-encoded certificate + private key identifying a node.
///
/// Serves as a server identity (what a node presents on its Flight port) and, under
/// mTLS, as a client identity (what a node presents when fetching from a peer).
///
/// `Debug` is hand-written to **redact the key**. A derived one prints it, and this
/// struct is reachable from `TlsServerConfig`/`TlsClientConfig`, which a transport error
/// path or a `tracing` span could format at any time — putting a node's private key into
/// the log stream, where it long outlives the process and is usually shipped off-box. The
/// engine already takes this position elsewhere: an inline `hmac_sha256` key raises a
/// `SecurityWarning` precisely because it would reach "any plan log / profile / explain
/// output". A private key deserves at least that much.
#[derive(Clone)]
pub struct TlsIdentity {
    /// PEM-encoded certificate chain (leaf first).
    pub cert_pem: String,
    /// PEM-encoded private key for the leaf certificate.
    pub key_pem: String,
}

impl TlsIdentity {
    /// Build a [`TlsIdentity`] from PEM certificate and key strings.
    pub fn from_pem(cert_pem: impl Into<String>, key_pem: impl Into<String>) -> Self {
        Self {
            cert_pem: cert_pem.into(),
            key_pem: key_pem.into(),
        }
    }

    fn to_tonic(&self) -> Identity {
        Identity::from_pem(self.cert_pem.as_bytes(), self.key_pem.as_bytes())
    }
}

impl std::fmt::Debug for TlsIdentity {
    /// Shows the certificate's presence and the key's *length* only.
    ///
    /// The length rather than nothing at all, because the failure this is most often
    /// formatted for is a malformed or empty PEM, and "key_pem: 0 bytes" answers that
    /// without disclosing anything a holder of the file does not already know.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsIdentity")
            .field("cert_pem_bytes", &self.cert_pem.len())
            .field(
                "key_pem",
                &format_args!("<redacted, {} bytes>", self.key_pem.len()),
            )
            .finish()
    }
}

/// Server-side TLS for a node's Flight server.
#[derive(Clone, Debug)]
pub struct TlsServerConfig {
    /// The certificate + key this node presents to connecting peers.
    pub identity: TlsIdentity,
    /// PEM CA bundle a client's certificate must chain to. `Some` turns on **mTLS**:
    /// a peer with no certificate, or one this CA did not sign, is rejected at the
    /// handshake — the network-level analogue of the shuffle token. `None` encrypts the
    /// connection but does not authenticate the client (server-auth TLS only).
    pub client_ca_pem: Option<String>,
}

impl TlsServerConfig {
    /// Server-auth TLS: encrypt the connection with `identity`, without requiring a
    /// client certificate.
    pub fn new(identity: TlsIdentity) -> Self {
        Self {
            identity,
            client_ca_pem: None,
        }
    }

    /// Require and verify client certificates against `client_ca_pem` (mTLS).
    pub fn with_client_ca(mut self, client_ca_pem: impl Into<String>) -> Self {
        self.client_ca_pem = Some(client_ca_pem.into());
        self
    }

    /// Translate to a tonic [`ServerTlsConfig`].
    pub(crate) fn to_tonic(&self) -> ServerTlsConfig {
        let mut cfg = ServerTlsConfig::new().identity(self.identity.to_tonic());
        if let Some(ca) = &self.client_ca_pem {
            cfg = cfg.client_ca_root(Certificate::from_pem(ca.as_bytes()));
        }
        cfg
    }
}

/// Client-side TLS for fetching from a peer's Flight server.
#[derive(Clone, Debug)]
pub struct TlsClientConfig {
    /// PEM CA bundle the peer's server certificate must chain to.
    pub ca_pem: String,
    /// The name to verify against the peer certificate's SAN. Peers are dialed by
    /// address, so the certificate rarely matches the literal host; set this to the name
    /// the cluster's certificates actually carry (e.g. the service name).
    pub domain: String,
    /// This node's client identity, presented under **mTLS**. Required whenever the
    /// peer server sets [`TlsServerConfig::client_ca_pem`]; ignored otherwise.
    pub identity: Option<TlsIdentity>,
}

impl TlsClientConfig {
    /// Verify the peer against `ca_pem`, matching its certificate to `domain`.
    pub fn new(ca_pem: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            ca_pem: ca_pem.into(),
            domain: domain.into(),
            identity: None,
        }
    }

    /// Present `identity` to an mTLS peer.
    pub fn with_identity(mut self, identity: TlsIdentity) -> Self {
        self.identity = Some(identity);
        self
    }

    /// Translate to a tonic [`ClientTlsConfig`].
    pub(crate) fn to_tonic(&self) -> ClientTlsConfig {
        let mut cfg = ClientTlsConfig::new()
            .ca_certificate(Certificate::from_pem(self.ca_pem.as_bytes()))
            .domain_name(self.domain.clone());
        if let Some(id) = &self.identity {
            cfg = cfg.identity(id.to_tonic());
        }
        cfg
    }
}

/// Map an infallible-looking PEM/tonic TLS build error into a transport error.
///
/// tonic validates the PEM lazily (at connect/serve), so a malformed cert surfaces here
/// as a `tonic::transport::Error`; wrap it so callers see a `TransportError` like any
/// other transport failure rather than a bare tonic type.
pub(crate) fn tls_error(context: &str, err: tonic::transport::Error) -> TransportError {
    TransportError::Io(format!("{context}: {err}"))
}

/// A URI must use the `https` scheme for a TLS channel; upgrade a bare/`http` address.
pub(crate) fn https_uri(addr: &str) -> String {
    if let Some(rest) = addr.strip_prefix("http://") {
        format!("https://{rest}")
    } else if addr.contains("://") {
        addr.to_string()
    } else {
        format!("https://{addr}")
    }
}

/// Best-effort validation that returns early with a clear error on obviously-malformed
/// PEM, rather than letting it fail deep in the handshake.
pub(crate) fn check_pem(label: &str, pem: &str) -> TransportResult<()> {
    if pem.contains("-----BEGIN ") && pem.contains("-----END ") {
        Ok(())
    } else {
        Err(TransportError::Io(format!(
            "{label} is not PEM-encoded (missing BEGIN/END markers)"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A derived `Debug` prints the private key, and this struct is reachable from both
    /// TLS configs — so a transport error path or a `tracing` span formatting one puts a
    /// node's key into the log stream, where it outlives the process and is usually
    /// shipped off-box.
    #[test]
    fn debug_never_discloses_the_private_key() {
        let id = TlsIdentity::from_pem(
            "-----BEGIN CERTIFICATE-----\nMIIBcert\n-----END CERTIFICATE-----",
            "-----BEGIN PRIVATE KEY-----\nSUPERSECRETKEYMATERIAL\n-----END PRIVATE KEY-----",
        );

        for rendered in [
            format!("{id:?}"),
            format!("{:?}", TlsServerConfig::new(id.clone())),
        ] {
            assert!(
                !rendered.contains("SUPERSECRETKEYMATERIAL"),
                "the private key reached a Debug rendering: {rendered}",
            );
            assert!(!rendered.contains("BEGIN PRIVATE KEY"));
            assert!(
                rendered.contains("redacted"),
                "the redaction should be visible"
            );
        }
    }

    /// The length is kept deliberately: the failure this is usually formatted for is an
    /// empty or malformed PEM, and the byte count answers that without disclosing anything.
    #[test]
    fn debug_still_reports_enough_to_diagnose_an_empty_pem() {
        let empty = TlsIdentity::from_pem("", "");
        let rendered = format!("{empty:?}");
        assert!(rendered.contains("0 bytes"), "{rendered}");
    }
}
