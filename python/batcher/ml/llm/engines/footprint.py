"""What a model will occupy, read before the engine is built rather than discovered by it.

Every piece of advice in `parallelism` needs two numbers — how large the weights are and how
large the device is — and on the real path neither was available, so the branch that matters
most could not fire. A tensor-parallel degree too small for the model is not a slow run: the
engine downloads tens of gigabytes, initializes, and dies on an allocation, minutes after the
point where the answer was already knowable.

Both numbers are cheap to get and neither needs the weights. A model's `config.json` carries
its shape (layers, key/value heads, head dimension), and its weight footprint is the sum of
the shard sizes the repository already publishes — one metadata call, or a directory listing
for a local path. The device's size is a driver query.

The parsing is deliberately separated from the fetching: `shape_from_config` and
`weight_bytes_from_files` are pure functions over a dict and a list, so the naming variance
across model families — and there is a lot of it — is testable without a network or a GPU.
Everything that reaches outside the process returns `None` on any failure, because a
footprint nobody could read must degrade to the advice being silent rather than to a warning
about a model that was never inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.logging import note_suppressed

__all__ = [
    "ModelShape",
    "declared_context",
    "device_total_bytes",
    "model_shape",
    "model_weight_bytes",
    "shape_from_config",
    "weight_bytes_from_files",
]

#: Files that hold weights. `.safetensors` is the modern format and `.bin`/`.pt` the pickle
#: one; a repository carrying both publishes the same weights twice, which is why the sum is
#: taken per format rather than over everything that looks large.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")

#: Seconds to wait on the hub's metadata before giving up on the footprint.
_HUB_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class ModelShape:
    """The four numbers that decide a transformer's memory, read from its configuration.

    Attributes:
        layers: Transformer blocks.
        attention_heads: Attention heads per layer — what a tensor-parallel degree must
            divide.
        kv_heads: Key/value heads per layer. Equal to `attention_heads` for plain multi-head
            attention, and far smaller under grouped-query attention, where it is the figure
            the cache actually scales with.
        head_dim: Dimension of one head.
        hidden_size: Model dimension, for the all-reduce estimate.
        max_context: The model's own maximum context window, `0` when it declares none.
    """

    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    max_context: int = 0


def shape_from_config(config: dict) -> ModelShape | None:
    """A `ModelShape` from a parsed `config.json`, or `None` when it is not a transformer.

    Model families disagree on nearly every key, and the disagreement is not cosmetic: reading
    `num_attention_heads` where a model publishes `num_key_value_heads` overstates the cache by
    the grouped-query ratio, which on a modern model is a factor of eight. So each field is
    looked up under the spellings that occur in practice, and a missing `head_dim` is derived
    from the hidden size rather than guessed.

    Args:
        config: The model's configuration, as `config.json` publishes it. A nested
            `text_config` — how a vision-language model carries its language tower — is
            followed, because that tower is what holds the cache.

    Returns:
        The shape, or `None` when the layer count or the head count is missing, which means
        this is not a configuration the arithmetic applies to.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.engines.footprint import shape_from_config
            >>> shape = shape_from_config(
            ...     {
            ...         "num_hidden_layers": 32,
            ...         "num_attention_heads": 32,
            ...         "num_key_value_heads": 8,
            ...         "hidden_size": 4096,
            ...     }
            ... )
            >>> shape.kv_heads, shape.head_dim
            (8, 128)
    """
    if not isinstance(config, dict):
        return None
    # A vision-language model puts the language tower — the half that holds a KV cache — under
    # `text_config`, and its top level carries the vision encoder's much smaller numbers.
    inner = config.get("text_config")
    if isinstance(inner, dict) and _first_int(inner, ("num_hidden_layers", "n_layer")):
        config = inner
    layers = _first_int(config, ("num_hidden_layers", "n_layer", "num_layers"))
    heads = _first_int(config, ("num_attention_heads", "n_head", "num_heads"))
    if not layers or not heads:
        return None
    hidden = _first_int(config, ("hidden_size", "n_embd", "d_model")) or 0
    kv_heads = _first_int(config, ("num_key_value_heads", "num_kv_heads", "multi_query_group_num"))
    head_dim = _first_int(config, ("head_dim", "kv_channels")) or (hidden // heads if hidden else 0)
    return ModelShape(
        layers=layers,
        attention_heads=heads,
        # A model that publishes no grouped count is multi-head attention, where every
        # attention head carries its own key and value.
        kv_heads=kv_heads or heads,
        head_dim=head_dim,
        hidden_size=hidden,
        max_context=_first_int(config, ("max_position_embeddings", "n_positions", "seq_length"))
        or 0,
    )


def _first_int(config: dict, keys: tuple[str, ...]) -> int:
    """The first of `keys` present as a positive integer, or `0`."""
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def weight_bytes_from_files(files: dict[str, int]) -> int:
    """Resident weight bytes implied by a repository's `{filename: size}` listing.

    A repository routinely publishes the same weights twice — `.safetensors` for the modern
    loader and `.bin` for the old one — so summing everything that looks like a weight file
    doubles the footprint and turns correct advice into a warning that a model needing one GPU
    needs two. The formats are summed separately and the largest single format wins, which is
    what a loader will actually read.

    Args:
        files: Weight-file names against their sizes in bytes.

    Returns:
        Bytes the weights occupy, `0` when no weight file is listed.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.engines.footprint import weight_bytes_from_files
            >>> weight_bytes_from_files(
            ...     {"model-00001.safetensors": 10, "model-00002.safetensors": 6, "x.bin": 16}
            ... )
            16
    """
    totals: dict[str, int] = {}
    for name, size in files.items():
        for suffix in _WEIGHT_SUFFIXES:
            if name.endswith(suffix) and isinstance(size, int) and size > 0:
                totals[suffix] = totals.get(suffix, 0) + size
                break
    return max(totals.values(), default=0)


def model_weight_bytes(model: str) -> int | None:
    """The weight footprint of `model`, from a local directory or the hub's metadata.

    Neither route downloads a weight. A local path is a directory listing; a repository id is
    one metadata call against the hub, which is a fraction of a second against the tens of
    minutes a wrong tensor-parallel degree costs by failing after the download.

    Args:
        model: A model id or a local path, exactly as it is handed to the engine.

    Returns:
        Bytes, or `None` when the model cannot be inspected — no such directory, no hub
        client, no network, a private repository. `None` means the advice stays silent, which
        is the only safe reading of "not measured".
    """
    local = _local_weight_bytes(model)
    if local is not None:
        return local
    try:
        from huggingface_hub import HfApi

        # Bounded, because this runs on the path to building an engine: an unreachable hub —
        # an air-gapped cluster, a proxy that blackholes rather than refuses — must cost a few
        # seconds of advice, not the worker's startup.
        info = HfApi().model_info(model, files_metadata=True, timeout=_HUB_TIMEOUT_S)
        sizes = {f.rfilename: int(f.size or 0) for f in (info.siblings or ())}
    except Exception as exc:
        note_suppressed("ml", "read the model's weight footprint from the hub", exc)
        return None
    return weight_bytes_from_files(sizes) or None


def _local_weight_bytes(model: str) -> int | None:
    """The footprint of a local model directory, or `None` when `model` is not one."""
    from pathlib import Path

    try:
        path = Path(model)
        if not path.is_dir():
            return None
        sizes = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}
    except OSError as exc:
        note_suppressed("ml", "read the model directory's weight footprint", exc)
        return None
    return weight_bytes_from_files(sizes) or None


def model_shape(model: str) -> ModelShape | None:
    """The shape of `model`, read from its configuration without loading it.

    Args:
        model: A model id or a local path.

    Returns:
        The shape, or `None` when the configuration cannot be read or is not a transformer's.
    """
    config = _read_config(model)
    return None if config is None else shape_from_config(config)


def _read_config(model: str) -> dict | None:
    """`config.json` for `model`, as a plain dict, or `None` when it cannot be read.

    Read as JSON rather than through `AutoConfig`. Two reasons, and both matter here. A model
    that ships its own configuration class needs `trust_remote_code=True` to load through
    `transformers`, which executes code from the repository — far too much to spend on reading
    six integers, and not something a sizing check should ever decide to do on a user's behalf.
    And with `trust_remote_code=False` those models simply fail, which is silent: the advice
    degrades to nothing on exactly the families that most often need a group.
    """
    import json
    from pathlib import Path

    try:
        local = Path(model) / "config.json"
        if local.is_file():
            return json.loads(local.read_text())
    except (OSError, ValueError) as exc:
        note_suppressed("ml", "read the model directory's configuration", exc)
        return None
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "config.json", etag_timeout=_HUB_TIMEOUT_S)
        return json.loads(Path(path).read_text())
    except Exception as exc:
        note_suppressed("ml", "read the model's configuration from the hub", exc)
        return None


def declared_context(model: str) -> int | None:
    """The context window `model` declares in its configuration, or `None` when it declares none.

    The bound a data-sized window must not exceed. Without it, a corpus of long prompts
    proposes a window above the model's maximum, vLLM refuses it, and the refusal is only
    discovered by *building the engine* — the weights load, the allocation happens, and the
    error arrives minutes later, after which the fallback build pays for all of it a second
    time. The number is in `config.json`, which costs a metadata read.

    Args:
        model: A model id or a local path.

    Returns:
        The declared window in tokens, or `None` when the configuration is unreadable or
        declares no maximum — where capping against a guess would be worse than not capping.
    """
    shape = model_shape(model)
    return shape.max_context if shape is not None and shape.max_context > 0 else None


def device_total_bytes() -> int | None:
    """This worker's smallest visible device memory in bytes, or `None` when unreadable.

    The *smallest* rather than the first: a group is bounded by its smallest member, and a
    heterogeneous node — an inference card beside a training card — is common enough on a
    shared cluster that taking device zero's size would advise a group that half the devices
    cannot hold.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        sizes = [
            int(torch.cuda.get_device_properties(i).total_memory)
            for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # pragma: no cover - no driver, no device, or an older torch
        note_suppressed("ml", "read the device memory size", exc)
        return None
    return min(sizes) if sizes else None
