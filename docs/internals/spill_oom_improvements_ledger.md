
### #39 — The spill codec finds a blob inside nesting

- **Was:** `SpillCodec::classify` matched only a column's **top-level** type
  (`Binary | LargeBinary | LargeUtf8`). A media scan's payload is a flat `LargeBinary`
  column, which it caught — but the shape the *spill path* produces is nested.
- **The case it missed:** `array_agg` over a blob column collects each group's values into a
  `List`, so the spilled partial state is `List<LargeBinary>` — and `array_agg` and
  `histogram` are precisely the aggregates that force the grace path rather than a bounded
  one. A struct column from a nested Parquet or JSON read is the same shape. Those are the
  spills where the payload most dwarfs the CPU, which is the whole justification for
  compressing, and they were going out uncompressed.
- **Now:** the check walks lists, structs, maps, unions, dictionaries, and run-end encoding.
  `Utf8` stays excluded deliberately — the measurement behind this policy is that compressing
  ordinary string state is a net *loss* on fast spill disk.
- **`FixedSizeBinary` is admitted only past 64 bytes.** The type spans a payload (an
  embedding or thumbnail, hundreds of bytes and up, compresses well) and an identifier (a
  16-byte UUID, a 4-byte address, a 32-byte digest, which does not). Compressing a schema
  because it carries a UUID would break the policy's "never a regression" property.
- **Proof:** `auto_codec_finds_a_blob_inside_nesting` (the `array_agg` state shape, a
  struct-nested blob, and a `List<Int64>` that must *not* trigger it, or nesting alone would
  have become the trigger) and `auto_codec_treats_only_wide_fixed_size_binary_as_a_blob`
  across six widths.
