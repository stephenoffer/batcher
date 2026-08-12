"""The scalar-function vocabulary — the documented home for `fn` discriminators.

`ir_tags.py` centralizes the *node* tags (the ``"e"`` discriminator). This module
is its sibling for the next level down: the ``fn`` string a function node carries
(``StrFunc(fn="contains")``, ``MathExpr(fn="sqrt")``). These strings mirror the Rust
``match`` arms in ``bc-expr``; keeping them named in one place turns a typo into a
clear `PlanError` at plan-build time instead of an opaque engine error, and gives
tooling/docs a single enumerable source of what the engine supports.

Two shapes, chosen by how the family grows:

* **Closed families** (a handful of stable operations) are `enum.StrEnum`s —
  ``MapFn``, ``ListBinaryFn``, ``ListSetFn``, ``Math2Fn``. The members read as code
  and a typo is an ``AttributeError``.
* **Open families** (string/math/date/list functions, which grow toward hundreds)
  are `frozenset`s — ``STR_FNS``, ``MATH_FNS``, ``DATE_FNS``, ``LIST_FNS``. A
  thousand-member ``Enum`` class would itself be the sprawl this codebase avoids; a
  named set is the scalable vocabulary, and adding a function is one new entry.

Every set/enum here is the *complete* vocabulary for its family (validated by the
test suite): the node base validates a node's ``fn`` against it at construction, so
the sets must stay exhaustive. Add the function's name here in the same change that
adds the namespace method and the Rust ``match`` arm.

The window-function sets live in `ir_tags` (the relational `Window` operator owns
them); they are re-exported here so this module is the one-stop view of the callable
vocabulary.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from batcher.plan.ir_tags import WINDOW_AGGREGATES, WINDOW_FUNCS, WINDOW_RANKING, WINDOW_VALUE

__all__ = [
    "AUDIO_FNS",
    "DATE_FNS",
    "GEO_FNS",
    "IMAGE_FNS",
    "KEYED_STR_FNS",
    "LIST_FNS",
    "MAKE_TEMPORAL_FNS",
    "MATH_FNS",
    "SEQ_FNS",
    "SPATIAL_FNS",
    "STR_FNS",
    "VIDEO_FNS",
    "WINDOW_AGGREGATES",
    "WINDOW_FUNCS",
    "WINDOW_RANKING",
    "WINDOW_VALUE",
    "ListBinaryFn",
    "ListSetFn",
    "MapFn",
    "Math2Fn",
]


class MapFn(StrEnum):
    """Map-column accessors carried by `MapFunc` (the `.map` namespace)."""

    MAP_KEYS = "map_keys"
    MAP_VALUES = "map_values"
    MAP_ENTRIES = "map_entries"
    ELEMENT_AT = "element_at"


class ListBinaryFn(StrEnum):
    """Pairwise reductions over two List columns carried by `ListBinary`."""

    DOT = "dot"
    COSINE_SIMILARITY = "cosine_similarity"
    L2_DISTANCE = "l2_distance"
    L1_DISTANCE = "l1_distance"
    HAMMING = "hamming"
    JACCARD = "jaccard"
    MULTISET_OVERLAP = "multiset_overlap"
    LCS_LENGTH = "lcs_length"


class ListSetFn(StrEnum):
    """Set operations between two List columns carried by `ListSet`."""

    ARRAY_INTERSECT = "array_intersect"
    ARRAY_EXCEPT = "array_except"
    ARRAY_UNION = "array_union"
    # `array_concat` rides this family because its shape is the same (two lists in, one
    # list out), but it is not a set operation: it appends without deduplicating.
    ARRAY_CONCAT = "array_concat"
    # `array_gather` rides this family for the same reason: two lists in, one list out. It
    # reads the right list as *positions* into the left, which is what makes `arg_sort` usable.
    ARRAY_GATHER = "array_gather"


class ListZipFn(StrEnum):
    """Element-wise arithmetic between two List columns carried by `ListZip`."""

    LIST_ADD = "list_add"
    LIST_SUBTRACT = "list_subtract"
    LIST_MULTIPLY = "list_multiply"


class Math2Fn(StrEnum):
    """Two-argument math functions carried by `Math2Expr` (→ Float64)."""

    POW = "pow"
    ATAN2 = "atan2"
    HYPOT = "hypot"
    GCD = "gcd"
    LCM = "lcm"
    ROUND = "round"  # round(x, digits)
    NEXT_AFTER = "next_after"


# --- Open families: named, exhaustive vocabularies (one entry per function) ------

STR_FNS: Final[frozenset[str]] = frozenset(
    {
        "aes_decrypt", "aes_encrypt", "ascii", "base64", "bit_length", "chunk", "contains",
        "compress", "crc32", "mime_type", "decompress", "ends_with", "from_base64", "hash64", "hex",
        "hmac_sha256", "ilike",
        "initcap", "json_array_length", "json_array_values", "json_exists",
        "json_extract_bool", "json_extract_float", "json_extract_int",
        "json_object_keys", "json_type", "json_value", "json_contains", "json_pretty",
        "json_structure", "chr", "to_base", "format_bytes", "format_bytes_si",
        "damerau_levenshtein", "jaro_similarity", "jaro_winkler_similarity",
        "json_extract_string", "l_trim",
        "len", "levenshtein", "like", "lower",
        "lpad", "mask", "md5", "minhash", "octet_length", "overlay", "position", "r_trim",
        "regexp_count", "regexp_extract", "regexp_extract_all", "regexp_matches",
        "regexp_replace", "regexp_replace_all", "regexp_split", "repeat", "replace",
        "reverse",
        "right", "rpad", "sha1", "sha256", "soundex", "split", "split_part",
        "starts_with", "strip_html", "substr", "substring_index", "to_case",
        "squad_normalize", "token_ngrams", "translate",
        # Per-document text quality — the Gopher/C4/RefinedWeb pretraining-corpus
        # filters, as per-row measures. The corpus-level *aggregates* of the same
        # properties live in `plan/functions/metrics/text/`; these answer "which
        # documents do I drop", which is what a filter needs.
        "word_count", "mean_word_length", "symbol_ratio",
        "alpha_word_ratio", "stopword_count", "bullet_line_ratio",
        "ellipsis_line_ratio", "duplicate_line_ratio", "duplicate_paragraph_ratio",
        "top_ngram_ratio", "duplicate_ngram_ratio", "char_entropy",
        "trim", "unhex", "upper", "xxhash64",
        "from_binary", "hamming", "jaccard_similarity", "parse_dirname", "parse_dirpath",
        "parse_filename", "parse_path", "regexp_escape", "to_binary", "url_decode",
        "url_encode",
    }
)  # fmt: skip

KEYED_STR_FNS: Final[frozenset[str]] = frozenset({"aes_decrypt", "aes_encrypt", "hmac_sha256"})
"""The `STR_FNS` members whose ``pattern`` slot carries secret key material.

Named here — rather than inline in the two places that care — because both the
plan-build-time key validation (`plan.functions.security`) and the `StrFunc.__repr__`
redaction must agree on the set exactly. A function added to one and not the other
would leak a key into a traceback."""

DATE_FNS: Final[frozenset[str]] = frozenset(
    {
        "century", "day", "day_of_week", "day_of_year", "dayname", "days_in_month",
        "decade", "epoch", "hour", "is_leap_year", "iso_year", "isodow", "last_day",
        "millennium", "minute", "month", "monthname", "quarter", "second", "week",
        "year",
    }
)  # fmt: skip

MAKE_TEMPORAL_FNS: Final[frozenset[str]] = frozenset(
    {
        "make_date", "make_timestamp", "from_unix_seconds", "from_unix_millis",
        "from_unix_micros", "from_unix_nanos", "from_unix_date",
    }
)  # fmt: skip
"""Temporal *constructors* carried by `MakeTemporal` — the inverse of `DATE_FNS`.

The epoch conversions are a family rather than one `from_epoch` node because the unit
is not a value the engine can infer: an Int64 column of epoch counts carries no record
of whether it counts seconds or nanoseconds, so the plan has to say."""

LIST_FNS: Final[frozenset[str]] = frozenset(
    {
        "arg_max", "arg_min", "arg_sort", "cum_sum", "diff", "entropy", "flatten", "l1_norm",
        "l2_norm", "len", "log_softmax",
        "max", "max_abs", "mean", "median", "min", "n_unique", "normalize", "product",
        "reverse", "softmax", "sort", "sort_desc", "std", "sum", "unique", "var",
    }
)  # fmt: skip

GEO_FNS: Final[frozenset[str]] = frozenset(
    {
        # Constructors and codecs.
        "st_point", "st_point_z", "st_make_line", "st_make_polygon", "st_make_envelope",
        "st_geom_from_text", "st_geom_from_wkb", "st_geom_from_geojson",
        "st_geom_from_geohash", "st_as_text", "st_as_ewkt", "st_as_binary", "st_as_ewkb",
        "st_as_hex_wkb", "st_as_geojson",
        # Accessors.
        "st_x", "st_y", "st_z", "st_xmin", "st_ymin", "st_xmax", "st_ymax",
        "st_geometry_type", "st_dimension", "st_srid", "st_set_srid", "st_num_points",
        "st_num_geometries", "st_num_interior_rings", "st_geometry_n", "st_point_n",
        "st_start_point", "st_end_point", "st_exterior_ring", "st_interior_ring_n",
        "st_is_empty", "st_is_valid", "st_is_valid_reason", "st_is_closed", "st_is_ring",
        "st_is_simple", "st_is_collection", "st_has_z", "st_coord_dim",
        # Measures — planar in coordinate units, then geodesic in metres.
        "st_area", "st_length", "st_perimeter", "st_distance", "st_max_distance",
        "st_hausdorff_distance", "st_azimuth", "st_distance_sphere",
        "st_distance_spheroid", "st_area_spheroid", "st_length_spheroid",
        "st_perimeter_spheroid",
        # Predicates.
        "st_intersects", "st_disjoint", "st_contains", "st_within", "st_covers",
        "st_covered_by", "st_touches", "st_crosses", "st_overlaps", "st_equals",
        "st_dwithin", "st_dwithin_sphere", "st_intersects_extent", "st_contains_extent",
        # Constructions and transforms.
        "st_centroid", "st_envelope", "st_boundary", "st_convex_hull",
        "st_point_on_surface", "st_buffer", "st_simplify", "st_reverse", "st_force2d",
        "st_force3d", "st_force_polygon_ccw", "st_force_polygon_cw",
        "st_flip_coordinates", "st_translate", "st_scale", "st_rotate", "st_affine",
        "st_snap_to_grid", "st_segmentize", "st_expand", "st_collect",
        "st_remove_repeated_points", "st_line_interpolate_point", "st_line_locate_point",
        "st_line_substring", "st_closest_point", "st_shortest_line", "st_project",
        "st_transform",
        # Grids and reference systems.
        "st_geohash", "geohash_encode", "geohash_decode_lon", "geohash_decode_lat",
        "st_tile_x", "st_tile_y", "st_quadkey", "st_s2_cell", "st_s2_cell_parent",
        "st_hex_bin", "st_hex_center_x", "st_hex_center_y", "st_utm_zone", "st_utm_epsg",
    }
)  # fmt: skip
"""The geospatial vocabulary, mirroring `bc_expr::GeoFunc`'s serde tags exactly.

Names follow PostGIS so a ported query reads the same, with one deliberate exception:
`st_force2d` and `st_force3d` carry no underscore before the digit, because that is what
Rust's `snake_case` rename produces for `StForce2d` and the wire contract is what the
engine actually deserializes. PostGIS spells them the same way.

The argument *count* is not stated here — it lives in `bc_expr::GeoFunc::arity`, which is
the one place that can check it, and the Python constructors in `plan/functions/geo/`
each build a fixed argument list so a mismatch is impossible to write."""

SPATIAL_FNS: Final[frozenset[str]] = frozenset(
    {
        # Quaternion properties. One quaternion in, one number out.
        "quat_norm", "quat_normalize_x", "quat_normalize_y", "quat_normalize_z",
        "quat_normalize_w", "quat_inverse_x", "quat_inverse_y", "quat_inverse_z",
        "quat_inverse_w", "quat_angle", "quat_to_roll", "quat_to_pitch", "quat_to_yaw",
        # Building a quaternion from the two representations a log is likely to carry.
        "quat_from_euler_x", "quat_from_euler_y", "quat_from_euler_z", "quat_from_euler_w",
        "quat_from_rotmat_x", "quat_from_rotmat_y", "quat_from_rotmat_z",
        "quat_from_rotmat_w",
        # Composing and comparing two rotations.
        "quat_multiply_x", "quat_multiply_y", "quat_multiply_z", "quat_multiply_w",
        "quat_angular_distance",
        # Interpolating between two rotations.
        "quat_slerp_x", "quat_slerp_y", "quat_slerp_z", "quat_slerp_w",
        # Rotating a vector.
        "quat_rotate_x", "quat_rotate_y", "quat_rotate_z", "quat_inverse_rotate_x",
        "quat_inverse_rotate_y", "quat_inverse_rotate_z",
        # Applying a full pose to a point.
        "se3_transform_x", "se3_transform_y", "se3_transform_z",
        "se3_inverse_transform_x", "se3_inverse_transform_y", "se3_inverse_transform_z",
    }
)  # fmt: skip
"""The rigid-body vocabulary, mirroring `bc_expr::SpatialFunc`'s serde tags exactly.

One name per *output component*, because a rotation produces three numbers and a
quaternion four. See `bc_expr::SpatialFunc` for why that beats a struct-returning node,
and `bc_spatial` for the component order and the angle conventions.

As with `GEO_FNS` the argument count is not stated here; it lives in
`bc_expr::SpatialFunc::arity`, and the Python constructors in
`plan/functions/spatial/` each build a fixed argument list."""

MATH_FNS: Final[frozenset[str]] = frozenset(
    {
        "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "bit_count",
        "cbrt", "ceil", "cos", "cosh", "cot", "degrees", "exp", "factorial", "floor",
        "ln", "log10", "log2", "radians", "round", "sign", "sin", "sinh", "sqrt",
        "tan", "tanh", "trunc", "csc", "even", "gamma", "lgamma", "rint", "sec",
    }
)  # fmt: skip

# --- Multimodal decode families ------------------------------------------------------
#
# These three were the vocabularies with no named home, and the omission had a cost:
# `IRNode.vocab` validates a node's `fn` at construction, so a family without a set here
# is a family where a typo becomes an opaque engine error at execution rather than a
# `PlanError` at plan build — and, because `tools/lint_ir_contract.py` derives its checks
# from these same sets, a family the engine and the control plane could drift apart on
# with nothing to notice.

IMAGE_FNS: Final[frozenset[str]] = frozenset(
    {
        "decode", "to_tensor", "to_tensor_f32", "to_grayscale", "center_crop",
        "resize", "encode", "convert", "dhash", "brightness", "sharpness",
        "auto_orient", "exif_orientation", "thumbnail", "letterbox",
    }
)  # fmt: skip
"""The `.image` vocabulary, mirroring `bc_expr::ImageFunc`'s serde tags exactly."""

AUDIO_FNS: Final[frozenset[str]] = frozenset(
    {
        "decode", "to_waveform", "resample", "trim_silence", "peak_normalize",
        "zero_crossing_rate", "mel_spectrogram", "mfcc",
    }
)  # fmt: skip
"""The `.audio` vocabulary, mirroring `bc_expr::AudioFunc`'s serde tags exactly."""

VIDEO_FNS: Final[frozenset[str]] = frozenset({"decode", "frames", "thumbnail", "frame_at"})
"""The `.video` vocabulary, mirroring `bc_expr::VideoFunc`'s serde tags exactly."""

SEQ_FNS: Final[frozenset[str]] = frozenset(
    {
        # Nucleotide transforms and composition.
        "complement", "reverse_complement", "transcribe", "back_transcribe",
        "gc_content", "gc_skew", "base_counts", "max_homopolymer", "is_valid",
        # Coding.
        "translate",
        # Sketching.
        "kmers", "canonical_kmers", "minimizers",
        # Physical chemistry.
        "melting_temp", "molecular_weight", "gravy", "isoelectric_point",
        # FASTQ quality.
        "phred_quality", "mean_quality", "expected_errors",
        # Motif search.
        "find_motif", "count_motif",
    }
)  # fmt: skip
"""The `.seq` vocabulary, mirroring `bc_expr::SeqFunc`'s serde tags exactly.

Names follow Biopython where Biopython has one (`reverse_complement`, `transcribe`,
`translate`, `gc_skew`, `molecular_weight`, `isoelectric_point`), because that is the
vocabulary a bioinformatician already has, and a ported script should read the same. Where it
does not, the name comes from the tool that owns the concept: `minimizers` from minimap2,
`canonical_kmers` from Jellyfish and KMC, `expected_errors` from the USEARCH/VSEARCH
`fastq_maxee` filter, `gravy` from the Kyte-Doolittle literature."""
