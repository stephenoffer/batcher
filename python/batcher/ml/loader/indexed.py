"""The indexed loaders: a deterministic, balanced, resumable global sample order per rank.

Both entry points here wrap `streaming_sampler`'s ordering contract in a
``torch.utils.data.IterableDataset``. They differ only in memory regime — `stream_loader`
streams the corpus once and retains this rank's shard of it (see `sharding`);
`shard_stream_loader` never materializes anything, computing its indices one batch at a time
from the keyed permutation. Same sample order either way.

Both stride their batch sequence across DataLoader workers (`_worker_stride`); an
`IterableDataset` that does not is replayed *in full* by every worker, so an epoch silently sees
each sample `num_workers` times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.prefetch import prefetch
from batcher.ml.converters import _worker_stride
from batcher.ml.loader.sharding import gather_rank_shard
from batcher.ml.loader.tensors import tensorize, warn_dropped_columns
from batcher.ml.stats._shared import require_names
from batcher.ml.streaming_sampler import num_rank_batches, rank_index_batches

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["shard_stream_loader", "stream_loader"]


def _warn_if_block_exceeds_cache(block: int | None, rows_per_shard: int, cache_size: int) -> None:
    """Warn when the shuffle window is wider than the shard cache can hold.

    The two knobs are one decision wearing two hats: a shuffle block spanning *k* shards needs
    *k* cache slots, or the loader evicts a shard it is about to read again and the epoch
    re-reads the corpus. Nothing here can adjust `shuffle_block_size` to fit — the sample order
    must not depend on a caching knob, or two ranks configured differently would train on
    different orders — so this says so instead of silently choosing.
    """
    import warnings

    needed = float("inf") if block is None else -(-block // max(1, rows_per_shard))
    if needed <= cache_size:
        return
    wanted = "a global shuffle" if block is None else f"a shuffle block of {block} samples"
    warnings.warn(
        f"{wanted} spans more shards than cache_size={cache_size} holds, so the reader will "
        f"evict shards it is about to read again and the epoch re-reads the corpus. Either "
        f"raise cache_size (to at least {needed if block is not None else 'the shard count'}) "
        f"or lower shuffle_block_size.",
        UserWarning,
        stacklevel=3,
    )


class _EpochState:
    """The ``set_epoch`` / ``state_dict`` protocol both indexed loaders expose.

    Two conventions a training loop already knows, and neither was reachable here before:
    `torch.utils.data.DistributedSampler.set_epoch`, which every DDP loop calls once per
    epoch, and MosaicML `StreamingDataset`'s ``state_dict``/``load_state_dict``, which is how
    a run resumes mid-epoch. Batcher's ordering has always *supported* both — the order is
    keyed on ``(seed, epoch)`` and seeks to any ``global_consumed`` in O(1) — but a caller had
    to thread the two numbers by hand and build a new loader for every epoch, so the feature
    was a protocol the user assembled rather than one the loader offered.

    Consumption is counted in **global** samples (``batch_size * world_size`` per yielded
    batch), which is the unit `global_consumed` is defined in and the unit that lets a job
    resume on a differently sized cluster.
    """

    __slots__ = ()

    def _corpus_rows(self) -> int:
        """Rows in the corpus this loader reads — half of the checkpoint's identity."""
        raise NotImplementedError

    def _corpus_seed(self) -> int:
        """The seed keying this loader's order — the other half."""
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        """Start `epoch`: reshuffle the global order and rewind to its first sample.

        Call once per epoch, on every rank, with the same value — the `DistributedSampler`
        contract. Ranks passing different values stride different permutations, so the epoch
        both repeats and skips samples.

        Args:
            epoch: The epoch to start. The same value on every rank.
        """
        self._set_position(epoch, 0)

    def state_dict(self) -> dict[str, Any]:
        """The resume point: the epoch, how far into it this run has read, and from what.

        Take it **between steps**, where every rank has consumed the same count, so
        `global_consumed` is a multiple of `world_size` and the resumed ranks stay balanced.

        Read it on the object you are iterating. Under ``DataLoader(num_workers=k)`` the
        loader is pickled into *k* separate processes, so the copy left in the parent never
        advances and would report a resume point of zero — checkpoint from a loop iterating
        this object directly, or with ``num_workers=0``.

        `num_samples` and `seed` ride along so `load_state_dict` can refuse a checkpoint the
        order would not survive. `world_size` deliberately does not: the global order is
        independent of it, which is the whole point of being able to resume on a
        differently-sized cluster.

        Returns:
            A mapping to hand back to `load_state_dict`.
        """
        epoch, consumed = self._position()
        return {
            "epoch": epoch,
            "global_consumed": consumed,
            "num_samples": self._corpus_rows(),
            "seed": self._corpus_seed(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Resume from a `state_dict`, mid-epoch, with nothing repeated and nothing skipped.

        The checkpoint is checked against this loader first. Resuming a run against a corpus
        that has since grown, or under a different `seed`, keeps the *position* meaningful
        and makes the *order* something else entirely — so the samples "already seen" are not
        the ones the previous run saw, and the epoch silently both repeats and skips. That is
        invisible in a loss curve and expensive to find, so it is refused rather than
        detected later. `ResumableSampler.load_state_dict` refuses on the same two fields.

        Args:
            state: A `state_dict` taken from a loader over the same corpus and seed.

        Raises:
            PlanError: If `state` describes a different corpus size or a different seed.
        """
        for key, current in (("num_samples", self._corpus_rows()), ("seed", self._corpus_seed())):
            recorded = state.get(key)
            if recorded is not None and recorded != current:
                raise PlanError(
                    f"cannot resume: the checkpoint has {key}={recorded!r} but this loader "
                    f"has {current!r}; the sample order would differ, so the samples it "
                    "records as consumed are not the ones this loader would have read"
                )
        self._set_position(int(state["epoch"]), int(state["global_consumed"]))


def _check_rank(world_size: int, rank: int) -> None:
    """Validate the data-parallel placement at construction, not on the first batch.

    A typo'd `rank`/`world_size` used to build a loader happily and raise from the index
    arithmetic on the *first iteration* — which in a DDP job is after every rank has
    initialized its process group. One rank raising there while the others wait at the
    all-reduce is a hang, not a failure, and a hang is what a job spends an hour on before
    anyone reads the traceback.
    """
    if world_size < 1:
        raise PlanError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise PlanError(f"rank {rank} out of range for world_size {world_size}")


def _check_batch_size(batch_size: int) -> int:
    """Validate a loader `batch_size`, raising rather than silently coercing.

    The sampler primitives these loaders wrap already reject ``batch_size < 1``; a
    silent ``max(1, batch_size)`` here would instead turn a typo'd ``batch_size=0`` into
    an epoch of one-row batches — correct output, but catastrophically slow, with no
    error to point at.
    """
    if batch_size < 1:
        raise PlanError(f"batch_size must be >= 1, got {batch_size}")
    return batch_size


def stream_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    columns: list[str] | None = None,
    global_consumed: int = 0,
    collate_fn: Any = None,
    shuffle_block_size: int | None = None,
) -> Any:
    """A `torch.utils.data.IterableDataset` of ``{column: tensor}`` batches for `rank`.

    The corpus is read once as a batch stream and only **this rank's shard** is retained, so
    peak driver memory is ``num_rows / world_size`` rows rather than the whole corpus per rank.
    Past that, write the corpus with `batcher.io.formats.ml.write_shards` and use
    `shard_stream_loader`: bounded memory, identical sample order.

    The dataset is read twice — once for `Dataset.count` (answered from metadata where the
    engine can) and once for the rows — so it must be deterministic in row order, which is the
    same assumption the shuffled global order already makes.

    Examples:
        .. doctest::

            >>> from batcher.ml import stream_loader  # doctest: +SKIP
            >>> loader = stream_loader(ds, batch_size=32, world_size=8, rank=0)  # doctest: +SKIP
            >>> for tensors in loader:  # doctest: +SKIP
            ...     step(tensors["features"], tensors["label"])

    Args:
        dataset: a bounded Batcher `Dataset` (streamed once; only this rank's rows are kept).
        batch_size: rows per yielded batch.
        world_size: ranks in the data-parallel group.
        rank: this process's slot in that group.
        epoch: selects this epoch's global order (with `seed`).
        seed: keys the permutation.
        shuffle: iterate in corpus order instead when false.
        drop_last: drop the epoch's tail remainder, else keep it and pad back up to a whole
            number of ranks. Either way every rank yields the same batch count, so no rank
            stalls the DDP barrier.
        columns: subset to tensorize (default: all tensorizable columns).
        global_consumed: samples already processed this epoch (resume point).
        collate_fn: optional ``pyarrow.Table -> Any`` collation applied to each batch instead
            of the default tensorization. The escape hatch for columns that cannot become
            tensors (string labels/ids) and for ragged sequences needing padding and an
            attention mask.
        shuffle_block_size: shuffle within blocks of this many samples rather than across
            the whole corpus. The rows are already resident here, so this buys locality
            rather than I/O; it exists mainly so a run can reproduce the order
            `shard_stream_loader` would give the same corpus.

    Returns:
        A `torch.utils.data.IterableDataset` of this rank's ``{column: tensor}`` batches.
    """
    from batcher._internal.optional import require

    IterableDataset = require(
        "torch.utils.data",
        "IterableDataset",
        feature="stream_loader",
        provides="torch",
        extra="torch",
    )

    bs = _check_batch_size(batch_size)
    _check_rank(world_size, rank)
    if columns is not None:
        require_names(list(dataset.columns), *columns, hint="Pass an existing column.")
    keep = list(columns) if columns is not None else list(dataset.columns)
    warned = False

    total_rows = dataset.count()

    def _gather(for_epoch: int, consumed: int):
        """This rank's rows for `for_epoch`, and the order to read them back in.

        Eager, and in the parent process, so the DataLoader workers forked below inherit the
        shard instead of each re-reading the corpus.
        """
        return gather_rank_shard(
            dataset,
            num_rows=total_rows,
            world_size=world_size,
            rank=rank,
            epoch=for_epoch,
            seed=seed,
            shuffle=shuffle,
            drop_last=drop_last,
            global_consumed=consumed,
            shuffle_block_size=shuffle_block_size,
        )

    class _StreamLoader(IterableDataset, _EpochState):  # type: ignore[misc]
        def __init__(self) -> None:
            self._epoch = epoch
            self._consumed = global_consumed
            self._table, self._order = _gather(epoch, global_consumed)
            self._probe()

        def _probe(self) -> None:
            nonlocal warned
            if self._table.num_rows and not warned:
                warn_dropped_columns(self._table.slice(0, 1), keep, collate_fn)
                warned = True

        def _corpus_rows(self) -> int:
            return total_rows

        def _corpus_seed(self) -> int:
            return seed

        def _position(self) -> tuple[int, int]:
            return self._epoch, self._consumed

        def _set_position(self, new_epoch: int, consumed: int) -> None:
            """Re-gather this rank's rows for the requested position.

            Not free, unlike `shard_stream_loader`'s: a new epoch is a new permutation, so
            *which corpus rows belong to this rank* changes, and the rows have to be read
            again. That is inherent to holding the shard in memory — the alternative is to
            hold the whole corpus. Past the point where the re-read hurts, write the corpus
            with `Dataset.ml.write_shards` and use `shard_stream_loader`, where changing
            epoch is arithmetic.
            """
            if (new_epoch, consumed) == (self._epoch, self._consumed):
                return
            self._epoch, self._consumed = new_epoch, consumed
            self._table, self._order = _gather(new_epoch, consumed)

        def __len__(self) -> int:
            # The rank's total across all DataLoader workers, not one worker's share — which
            # is what a training loop wants for its step count.
            n = len(self._order)
            return n // bs if drop_last else (n + bs - 1) // bs

        def __iter__(self):
            import pyarrow as pa

            table, order = self._table, self._order
            n = len(order)
            limit = (n // bs) * bs if drop_last else n
            start_consumed = self._consumed
            per_batch = bs * world_size
            # Stride before `take`, so a worker pays no gather for a batch it does not own.
            offset, stride = _worker_stride()
            for i, first in enumerate(range(0, limit, bs)):
                if i % stride == offset:
                    rows = pa.array(order[first : first + bs])
                    self._consumed = start_consumed + (i + 1) * per_batch
                    yield tensorize(table.take(rows), keep, collate_fn)

    return _StreamLoader()


def shard_stream_loader(
    directory: str,
    *,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    columns: list[str] | None = None,
    global_consumed: int = 0,
    cache_size: int = 4,
    collate_fn: Any = None,
    shuffle_block_size: int | None = None,
    prefetch_batches: int = 2,
) -> Any:
    """A bounded-memory `IterableDataset` streaming a **shard directory** written by Batcher.

    Same sample-order contract as `stream_loader`, but the corpus is never materialized: the
    row count comes from the shard index (written by `batcher.io.formats.ml.write_shards`) and
    each batch is gathered through a `ShardReader` whose LRU cache keeps at most `cache_size`
    shards resident — so it scales past RAM, the case `stream_loader` can't handle.

    Examples:
        .. doctest::

            >>> from batcher.ml import shard_stream_loader  # doctest: +SKIP
            >>> loader = shard_stream_loader("/data/shards", batch_size=32)  # doctest: +SKIP
            >>> len(loader)  # batches this rank yields  # doctest: +SKIP

    Args:
        directory: the shard directory to stream (holds the shards and their index).
        batch_size: rows per yielded batch.
        world_size: ranks in the data-parallel group.
        rank: this process's slot in that group.
        epoch: selects this epoch's global order (with `seed`).
        seed: keys the permutation.
        shuffle: iterate in corpus order instead when false.
        drop_last: drop the epoch's tail remainder and any partial final batch.
        columns: subset to tensorize (default: all tensorizable columns).
        global_consumed: samples already processed this epoch (resume point).
        cache_size: shards kept resident in the reader's LRU cache.
        collate_fn: optional ``pyarrow.Table -> Any`` collation applied to each batch instead
            of the default tensorization, for columns that cannot become tensors and for
            ragged sequences needing padding and an attention mask.
        shuffle_block_size: shuffle within blocks of this many samples (and shuffle the
            blocks) rather than across the whole corpus. Defaults to one shard, which
            shuffles the shards and the rows inside each — the order is then a property of
            the corpus and `seed` alone, so every rank agrees however its cache is sized. A
            wider block decorrelates further and needs a proportionally larger `cache_size`;
            ``0`` restores a global shuffle, which needs the whole corpus cached to read
            each shard once.
        prefetch_batches: batches whose shard reads run ahead on a background thread, so
            the next batch's I/O overlaps this batch's training step. 0 disables it.

    Returns:
        A `torch.utils.data.IterableDataset` of this rank's ``{column: tensor}`` batches.
    """
    from batcher._internal.optional import require

    IterableDataset = require(
        "torch.utils.data",
        "IterableDataset",
        feature="stream_loader",
        provides="torch",
        extra="torch",
    )

    from batcher.io.formats.ml.shards import ShardReader

    reader = ShardReader(directory, cache_size=cache_size)
    _check_rank(world_size, rank)
    # The schema comes off the index, so the probe costs no shard read on a corpus written
    # by a current `write_shards`; a one-row `take` is only needed to see actual values.
    if columns is not None:
        require_names(list(reader.schema.names), *columns, hint="Pass an existing column.")
    keep = list(columns) if columns is not None else list(reader.schema.names)
    if reader.total_rows:
        warn_dropped_columns(reader.take([0]), keep, collate_fn)
    bs = _check_batch_size(batch_size)
    # The default is one shard wide: shuffle the shards, and shuffle the rows inside each —
    # WebDataset's and MosaicML's ``py1s``. It depends only on how the corpus was *written*,
    # never on `cache_size`, because a sample order that moved with a caching knob would put
    # two ranks on different orders the moment their caches were sized differently.
    block = (
        reader.index.rows_per_shard if shuffle_block_size is None else (shuffle_block_size or None)
    )
    if shuffle:
        _warn_if_block_exceeds_cache(block, reader.index.rows_per_shard, cache_size)
    # The larger-than-RAM path, so the *index* sequence must not be materialized either: a
    # shuffled list of every index costs ~28 bytes/sample (280 GB at 10^10 samples).
    # `rank_index_batches` computes this rank's indices one batch at a time from a keyed
    # permutation — identical order, O(batch_size) memory.
    sampler_kwargs: dict[str, Any] = {
        "batch_size": bs,
        "world_size": world_size,
        "rank": rank,
        "epoch": epoch,
        "seed": seed,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "global_consumed": global_consumed,
        "shuffle_block_size": block,
    }

    class _ShardLoader(IterableDataset, _EpochState):  # type: ignore[misc]
        def __init__(self) -> None:
            self._epoch = epoch
            self._consumed = global_consumed

        def _corpus_rows(self) -> int:
            return reader.total_rows

        def _corpus_seed(self) -> int:
            return seed

        def _position(self) -> tuple[int, int]:
            return self._epoch, self._consumed

        def _set_position(self, new_epoch: int, consumed: int) -> None:
            self._epoch, self._consumed = new_epoch, consumed

        def __len__(self) -> int:
            return num_rank_batches(
                reader.total_rows,
                batch_size=bs,
                world_size=world_size,
                rank=rank,
                drop_last=drop_last,
                global_consumed=self._consumed,
            )

        def __iter__(self):
            # Stride the *index* generator, so a worker never issues the shard read for a
            # batch another worker owns. `rank_index_batches` is O(1) memory, so advancing
            # past a skipped batch costs arithmetic, not I/O.
            offset, stride = _worker_stride()
            start_consumed = self._consumed
            kwargs = {**sampler_kwargs, "epoch": self._epoch, "global_consumed": start_consumed}
            per_batch = bs * world_size  # global samples one yielded batch accounts for

            def _read() -> Any:
                for i, indices in enumerate(rank_index_batches(reader.total_rows, **kwargs)):
                    if i % stride == offset:
                        # `take_batch`, not `take`: one contiguous batch, so the tensor
                        # conversion does not walk a chunk list per column.
                        yield i, reader.take_batch(indices)

            # The shard read is I/O and the tensorize is CPU, so overlapping them with the
            # consumer's training step is most of what a loader can do for a GPU here.
            for i, batch in prefetch(_read(), prefetch_batches):
                # Counted before the yield: a `state_dict` taken after receiving k batches
                # must say k were consumed, and post-yield code only runs once the consumer
                # asks for the next one — or never, if it stops. `i` is the batch's position
                # in the rank's whole sequence, so the count is right in a worker that only
                # produced every `stride`-th batch.
                self._consumed = start_consumed + (i + 1) * per_batch
                yield tensorize(batch, keep, collate_fn)

    return _ShardLoader()
