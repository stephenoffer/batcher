//! Options of the series window functions (`ewm_*`, `interpolate`) and of `qcut`.
//!
//! These ride on a [`crate::WindowFunc`] as one nested `opts` object rather than as
//! loose fields, because only those functions read them and every other window
//! function must see the defaults. The Python control plane omits the object when every
//! option is at its default, so a plan built before these existed deserializes unchanged.

use serde::{Deserialize, Deserializer};

/// The tuning knobs of the series recurrences beyond the EWM smoothing factor, and the
/// quantiles `qcut` bins by.
///
/// Every field defaults to the behaviour the functions had before it existed, so an absent
/// `opts` object (or an absent key inside one) changes nothing.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct WindowOpts {
    /// EWM: `true` divides by the decayed sum of every weight (pandas/Polars
    /// `adjust=True`); `false` is the recursive form `y_t = (1 - alpha)·y_{t-1} + alpha·x_t`
    /// (`adjust=False`).
    pub adjust: bool,
    /// EWM: the number of non-null observations a partition must have seen before a row
    /// gets a value; earlier rows are null (pandas/Polars `min_periods`).
    pub min_periods: u64,
    /// `interpolate`: a gap wider than this stays null. Measured in null rows by default,
    /// and as the order key's distance between the bracketing readings when
    /// [`Self::by_value`] is set (microseconds for a temporal key). `None` is no limit.
    pub max_gap: Option<f64>,
    /// `interpolate`: weight the straight line by the order key's *value* distance rather
    /// than by row position (Polars `interpolate_by`), for an irregularly sampled series.
    pub by_value: bool,
    /// `qcut`: the probabilities whose quantiles are the bin edges, strictly increasing in
    /// `[0, 1]`. Empty for every other function. Sent as decimal *strings* (see
    /// [`exact_floats`]), because the last bit of each one decides which bin a value lying
    /// exactly on an edge joins.
    #[serde(deserialize_with = "exact_floats")]
    pub probs: Vec<f64>,
    /// `qcut`: merge edges that tied input made equal instead of raising (pandas
    /// `duplicates="drop"`).
    pub drop_duplicates: bool,
}

/// Parse floats sent as their shortest round-trip decimal strings.
///
/// `serde_json` without its `float_roundtrip` feature parses a long JSON number to a
/// neighbouring double rather than the nearest one -- measured at about one value in ten for
/// probabilities such as `0.30000000000000004`. Rust's `str::parse::<f64>` is correctly
/// rounded, so a string Python wrote with `repr` comes back bit-identical, without paying the
/// slower exact parser for every number in every plan.
fn exact_floats<'de, D: Deserializer<'de>>(de: D) -> Result<Vec<f64>, D::Error> {
    let raw: Vec<String> = Vec::deserialize(de)?;
    raw.iter()
        .map(|s| s.parse::<f64>().map_err(serde::de::Error::custom))
        .collect()
}

impl Default for WindowOpts {
    fn default() -> Self {
        Self {
            adjust: true,
            min_periods: 1,
            max_gap: None,
            by_value: false,
            probs: Vec::new(),
            drop_duplicates: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty object and a partial one both keep the historical defaults for the keys
    /// they omit, which is what lets Python send only the non-default options.
    #[test]
    fn missing_keys_take_the_historical_defaults() {
        let empty: WindowOpts = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, WindowOpts::default());
        let partial: WindowOpts =
            serde_json::from_str(r#"{"adjust": false, "max_gap": 2.0}"#).unwrap();
        assert!(!partial.adjust);
        assert_eq!(partial.max_gap, Some(2.0));
        assert_eq!(partial.min_periods, 1);
        assert!(!partial.by_value);
    }

    /// Probabilities arrive as strings and parse to the exact double Python held.
    #[test]
    fn probabilities_parse_bit_exactly() {
        let opts: WindowOpts =
            serde_json::from_str(r#"{"probs": ["0", "0.30000000000000004", "1.0"]}"#).unwrap();
        assert_eq!(opts.probs, vec![0.0, 0.1 + 0.2, 1.0]);
        assert_eq!(opts.probs[1].to_bits(), (0.1f64 + 0.2).to_bits());
    }
}
