//! The structured shuffle coordinate ([`ShuffleTicket`]) the distributed layer
//! uses to build and parse the opaque ticket string carried on the wire.

use crate::{TransportError, TransportResult};

/// A structured shuffle coordinate identifying one output partition stream.
///
/// On the wire the ticket is an opaque string; this struct gives the
/// distributed layer a typed way to build and parse it. The string form is
/// `"<plan_id>/<stage_id>/<src_partition>/<dst_partition>/<epoch>"` (five
/// slash-separated unsigned integers), e.g. `"7/3/12/45/0"`.
///
/// `epoch` distinguishes re-executions of the same logical partition (e.g.
/// after a speculative retry or a recovered map task) so a stale producer's
/// output is never confused with a fresh one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShuffleTicket {
    /// Logical plan / query identifier.
    pub plan_id: u64,
    /// Shuffle stage within the plan.
    pub stage_id: u32,
    /// Source (map) partition that produced this output.
    pub src_partition: u32,
    /// Destination (reduce) partition this output feeds.
    pub dst_partition: u32,
    /// Re-execution epoch (0 for the first attempt).
    pub epoch: u32,
}

impl ShuffleTicket {
    /// Construct a ticket from its components.
    pub fn new(
        plan_id: u64,
        stage_id: u32,
        src_partition: u32,
        dst_partition: u32,
        epoch: u32,
    ) -> Self {
        Self {
            plan_id,
            stage_id,
            src_partition,
            dst_partition,
            epoch,
        }
    }

    /// Render the ticket as its canonical string form. (Mirrors the `Display`
    /// impl; provided as a named method per the shuffle-ticket API contract.)
    #[allow(clippy::inherent_to_string_shadow_display)]
    pub fn to_string(&self) -> String {
        format!(
            "{}/{}/{}/{}/{}",
            self.plan_id, self.stage_id, self.src_partition, self.dst_partition, self.epoch
        )
    }

    /// Parse a ticket from its canonical string form. Returns an error if the
    /// shape or any field fails to parse.
    pub fn from_string(s: &str) -> TransportResult<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 5 {
            return Err(TransportError::Io(format!(
                "invalid shuffle ticket {s:?}: expected 5 '/'-separated fields, got {}",
                parts.len()
            )));
        }
        let parse_u64 = |p: &str| {
            p.parse::<u64>()
                .map_err(|e| TransportError::Io(format!("invalid shuffle ticket {s:?}: {e}")))
        };
        let parse_u32 = |p: &str| {
            p.parse::<u32>()
                .map_err(|e| TransportError::Io(format!("invalid shuffle ticket {s:?}: {e}")))
        };
        Ok(Self {
            plan_id: parse_u64(parts[0])?,
            stage_id: parse_u32(parts[1])?,
            src_partition: parse_u32(parts[2])?,
            dst_partition: parse_u32(parts[3])?,
            epoch: parse_u32(parts[4])?,
        })
    }
}

impl std::fmt::Display for ShuffleTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ShuffleTicket::to_string(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_string_form() {
        let t = ShuffleTicket::new(7, 3, 12, 45, 0);
        assert_eq!(t.to_string(), "7/3/12/45/0");
        assert_eq!(ShuffleTicket::from_string("7/3/12/45/0").unwrap(), t);
    }

    #[test]
    fn display_matches_canonical_string() {
        let t = ShuffleTicket::new(1, 2, 3, 4, 5);
        assert_eq!(format!("{t}"), t.to_string());
    }

    #[test]
    fn large_plan_id_uses_full_u64_range() {
        let t = ShuffleTicket::new(u64::MAX, u32::MAX, 0, 0, u32::MAX);
        assert_eq!(ShuffleTicket::from_string(&t.to_string()).unwrap(), t);
    }

    #[test]
    fn epoch_distinguishes_reexecutions() {
        // Same logical partition, different attempt → different ticket (so a stale
        // producer's output is never confused with a fresh one).
        let first = ShuffleTicket::new(7, 3, 12, 45, 0);
        let retry = ShuffleTicket::new(7, 3, 12, 45, 1);
        assert_ne!(first, retry);
        assert_ne!(first.to_string(), retry.to_string());
    }

    #[test]
    fn wrong_field_count_is_rejected() {
        assert!(ShuffleTicket::from_string("7/3/12/45").is_err()); // 4 fields
        assert!(ShuffleTicket::from_string("7/3/12/45/0/9").is_err()); // 6 fields
        assert!(ShuffleTicket::from_string("").is_err());
    }

    #[test]
    fn non_numeric_field_is_rejected() {
        assert!(ShuffleTicket::from_string("7/x/12/45/0").is_err());
        assert!(ShuffleTicket::from_string("-1/3/12/45/0").is_err()); // u64 can't be negative
    }
}
