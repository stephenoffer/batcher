"""The `.seq` expression namespace — genomics and proteomics over a text column.

`SeqFunc` lowers to ``{"e": "seq", "fn": ...}`` IR consumed by Rust `Expr::Seq`. Every
per-base operation runs in the data plane, so a genome-scale scan never materializes a
sequence in Python.

A genomics pipeline is a string pipeline with a different alphabet, which is exactly why it
needs its own namespace rather than a composition of `.str` calls. Reverse-complement is not
``reverse`` plus ``translate``; codon translation reads three bases at a time and has no
substring spelling that is not a row loop; and GC content written as
``len(replace(s, 'A', '')) / len(s)`` allocates two strings per row *and* silently counts the
``N``s the caller meant to exclude.

Case is **preserved** by the transforms and **folded** by the measures. That is the Biopython
convention and it is load-bearing: lowercase is how every reference genome marks soft-masked
repeats, so a transform that upper-cased would destroy the mask, while a measure that respected
it would report a repeat-rich contig as mostly-unknown.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.fn_names import SEQ_FNS
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["SeqFunc", "_SeqNamespace"]

# The alphabets `is_valid` accepts; mirrors `bc-expr`'s `alphabet_bytes` exactly. Named here
# so a typo fails at plan build rather than nulling a column.
_ALPHABETS = frozenset({"dna", "rna", "dna_iupac", "rna_iupac", "protein"})

# The subset of those with a defined molecular mass. The degenerate alphabets are excluded
# because an ambiguity code has no mass, and returning a column of nulls would be a worse
# answer than refusing the question.
_WEIGHT_ALPHABETS = frozenset({"dna", "rna", "protein"})

# The largest k-mer length the engine accepts; mirrors `bc-expr`'s `MAX_K`. Assembly uses
# 21-127, alignment seeds 15-31, taxonomic classification 31-35 — 256 covers every real use,
# and beyond it the k-mer list is longer than the sequence it came from.
_MAX_K = 256


@expr_node
class SeqFunc(IRNode):
    """A biological-sequence op over a Utf8 sub-expression (via `.seq`).

    Its own node rather than more `StrFunc` arms because the argument shape genuinely differs:
    a string function carries a pattern and a window, while these carry a k-mer length, a
    reading frame, an ASCII quality offset, and an alphabet.
    """

    tag = ExprTag.SEQ
    vocab = SEQ_FNS
    fn: str = scalar()
    input: Expr = child()
    # Every argument is omitted from the IR unless set, so each function's wire shape carries
    # only what it actually uses.
    k: int | None = scalar(omit_none=True, default=None)
    window: int | None = scalar(omit_none=True, default=None)
    frame: int | None = scalar(omit_none=True, default=None)
    offset: int | None = scalar(omit_none=True, default=None)
    alphabet: str | None = scalar(omit_none=True, default=None)
    pattern: str | None = scalar(omit_none=True, default=None)
    to_stop: bool = scalar(omit_falsy=True, default=False)


def _check_alphabet(method: str, alphabet: str, allowed: frozenset[str]) -> None:
    """Reject an unknown alphabet at plan build, where the message can name the choices."""
    if alphabet not in allowed:
        raise PlanError(
            f"seq.{method}(): alphabet must be one of {sorted(allowed)}, got {alphabet!r}"
        )


def _check_k(method: str, k: int) -> None:
    """Reject a k-mer length the engine would refuse, with the same bounds it uses."""
    if not 1 <= k <= _MAX_K:
        raise PlanError(f"seq.{method}(): k must be in 1..={_MAX_K}, got {k}")


class _SeqNamespace:
    """Genomics and proteomics: ``col("seq").seq.reverse_complement()`` / ``.seq.gc_content()``.

    Every operation runs per-base in the Rust data plane over a text column, so a genome-scale
    scan never materializes a sequence in Python. Null input yields null.

    The namespace covers four families: nucleotide transforms and composition
    (:meth:`reverse_complement`, :meth:`gc_content`, :meth:`base_counts`), coding
    (:meth:`translate`), sketching (:meth:`kmers`, :meth:`minimizers`), and the physical
    properties a primer or protein is filtered on (:meth:`melting_temp`,
    :meth:`molecular_weight`, :meth:`isoelectric_point`), plus FASTQ quality decoding
    (:meth:`mean_quality`, :meth:`expected_errors`).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"seq": ["ATGGCC", "GGCCAT"]})
            >>> ds.select(rc=bt.col("seq").seq.reverse_complement()).to_pydict()
            {'rc': ['GGCCAT', 'ATGGCC']}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.seq` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.seq accessor of col('c')>``."""
        return f"<.seq accessor of {self._e!r}>"

    # --- Nucleotide transforms ------------------------------------------------------

    def complement(self) -> SeqFunc:
        """Replace each base with its IUPAC complement, keeping the reading direction.

        Case is preserved, because lowercase is how a reference genome marks soft-masked
        repeats and upper-casing would destroy the mask. The ambiguity codes complement as
        IUPAC defines them (``R``/``Y``, ``K``/``M``, ``B``/``V``, ``D``/``H`` swap; ``S``,
        ``W``, and ``N`` are their own complements), which is the half of this operation that
        is invisible on a test of pure ACGT and wrong on any real variant call.

        Any character that is not a nucleotide code passes through unchanged, so a gap
        character stays a gap character.

        Returns:
            An expression evaluating to the complemented sequence; null for null input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ATGC", "atgc", "RYSWN"]})
                >>> ds.select(c=bt.col("s").seq.complement()).to_pydict()
                {'c': ['TACG', 'tacg', 'YRSWN']}
        """
        return SeqFunc("complement", self._e)

    def reverse_complement(self) -> SeqFunc:
        """Read the opposite strand: the complement, 3' to 5'.

        The single most-used operation in genomics. A sequencing read maps to either strand,
        so comparing two reads, aligning a primer, or normalizing a k-mer all start here.
        Case and non-nucleotide characters are handled exactly as in :meth:`complement`.

        Returns:
            An expression evaluating to the reverse-complemented sequence; null for null
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ATGGCC"]})
                >>> ds.select(rc=bt.col("s").seq.reverse_complement()).to_pydict()
                {'rc': ['GGCCAT']}

                >>> # Reverse-complementing twice is the identity.
                >>> both = bt.col("s").seq.reverse_complement().seq.reverse_complement()
                >>> ds.select(x=both).to_pydict()
                {'x': ['ATGGCC']}
        """
        return SeqFunc("reverse_complement", self._e)

    def transcribe(self) -> SeqFunc:
        """Convert DNA to RNA by replacing ``T`` with ``U``, case preserved.

        Returns:
            An expression evaluating to the RNA sequence; null for null input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ATGGCC"]})
                >>> ds.select(r=bt.col("s").seq.transcribe()).to_pydict()
                {'r': ['AUGGCC']}
        """
        return SeqFunc("transcribe", self._e)

    def back_transcribe(self) -> SeqFunc:
        """Convert RNA to DNA by replacing ``U`` with ``T``, case preserved.

        Returns:
            An expression evaluating to the DNA sequence; null for null input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["AUGGCC"]})
                >>> ds.select(d=bt.col("s").seq.back_transcribe()).to_pydict()
                {'d': ['ATGGCC']}
        """
        return SeqFunc("back_transcribe", self._e)

    # --- Composition ----------------------------------------------------------------

    def gc_content(self) -> SeqFunc:
        """The G+C fraction of the unambiguous bases, in ``[0, 1]``.

        Ambiguous bases are excluded from the denominator rather than counted as non-GC. That
        is the difference between "no data" and "a real signal": a run of ``N`` counted as
        AT would put an assembly gap at the AT-rich extreme of every histogram and every
        filter built on one. ``U`` counts as a pyrimidine like ``T``, so a DNA and an RNA
        column give the same answer.

        Returns:
            An expression evaluating to a Float64 in ``[0, 1]``; null for null input and for a
            row with no unambiguous base (including the empty sequence).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["GCAT", "GCNN", "NNNN"]})
                >>> ds.select(gc=bt.col("s").seq.gc_content()).to_pydict()
                {'gc': [0.5, 1.0, None]}
        """
        return SeqFunc("gc_content", self._e)

    def gc_skew(self) -> SeqFunc:
        """``(G - C) / (G + C)``, the replication-strand signal, in ``[-1, 1]``.

        The leading and lagging strands of a replicating chromosome mutate differently, so the
        sign of this quantity flips at the replication origin and again at the terminus.
        Computed over a sliding window it is how an origin is located in a newly assembled
        bacterial genome.

        Returns:
            An expression evaluating to a Float64 in ``[-1, 1]``; null for null input and for a
            row containing no G or C.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["GGGC", "CCCG", "ATAT"]})
                >>> ds.select(skew=bt.col("s").seq.gc_skew()).to_pydict()
                {'skew': [0.5, -0.5, None]}
        """
        return SeqFunc("gc_skew", self._e)

    def base_counts(self) -> SeqFunc:
        """Count every base at once, as a struct ``{a, c, g, t, u, n, other}``.

        One pass yields every count, so asking for the ``N`` count costs nothing beyond asking
        for the ``A`` count. Case-folded. Project the one you want with
        ``.struct.field("n")``.

        ``n`` counts only the literal ``N`` code; the other IUPAC ambiguity codes and any
        non-nucleotide character land in ``other``, which is what makes ``other > 0`` a usable
        "this row is not clean sequence" predicate.

        Returns:
            An expression evaluating to a struct of seven Int64 counts; null for null input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["AACGTN"]})
                >>> ds.select(n=bt.col("s").seq.base_counts().struct.field("n")).to_pydict()
                {'n': [1]}

                >>> ds.select(b=bt.col("s").seq.base_counts()).to_pydict()
                {'b': [{'a': 2, 'c': 1, 'g': 1, 't': 1, 'u': 0, 'n': 1, 'other': 0}]}
        """
        return SeqFunc("base_counts", self._e)

    def max_homopolymer(self) -> SeqFunc:
        """The length of the longest run of a single base.

        The nanopore and PacBio error signature: those chemistries systematically miscount long
        single-base runs, so homopolymer length is what a variant filter thresholds on to
        separate a real indel from a sequencing artefact.

        Case-folded, so a run that crosses a soft-mask boundary counts as one run.

        Returns:
            An expression evaluating to an Int64 (0 for the empty sequence); null for null
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ACGT", "AaaaTG"]})
                >>> ds.select(h=bt.col("s").seq.max_homopolymer()).to_pydict()
                {'h': [1, 4]}
        """
        return SeqFunc("max_homopolymer", self._e)

    def is_valid(self, alphabet: str) -> SeqFunc:
        """Whether every character belongs to `alphabet`.

        The gate to put in front of everything else: a column that mixes sequence with headers,
        quality strings, or an empty-string sentinel fails here loudly rather than producing
        plausible nulls three operations later.

        The empty sequence is **valid** — it violates no membership rule, and conflating "has a
        bad character" with "has no characters" would hide two different problems behind one
        flag. Filter on length for the second.

        Args:
            alphabet: One of ``"dna"`` (ACGT), ``"rna"`` (ACGU), ``"dna_iupac"`` or
                ``"rna_iupac"`` (those plus the degenerate codes and ``N``), or ``"protein"``
                (the 20 standard residues plus the ``*`` stop marker :meth:`translate` emits).
                Case-insensitive in the data.

        Returns:
            An expression evaluating to a Boolean; null for null input.

        Raises:
            PlanError: If `alphabet` is not one of the five.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ACGT", "ACGN", "hello"]})
                >>> ds.select(ok=bt.col("s").seq.is_valid("dna")).to_pydict()
                {'ok': [True, False, False]}

                >>> ds.select(ok=bt.col("s").seq.is_valid("dna_iupac")).to_pydict()
                {'ok': [True, True, False]}
        """
        _check_alphabet("is_valid", alphabet, _ALPHABETS)
        return SeqFunc("is_valid", self._e, alphabet=alphabet)

    # --- Coding ---------------------------------------------------------------------

    def translate(self, *, frame: int = 0, to_stop: bool = False) -> SeqFunc:
        """Translate codons to amino acids using the standard genetic code.

        NCBI translation table 1, reading non-overlapping triplets from `frame`. Accepts DNA or
        RNA. A codon containing any ambiguous base becomes ``X``, a stop codon becomes ``*``,
        and a trailing partial codon is dropped rather than padded — padding it would fabricate
        a residue the data does not support.

        Args:
            frame: The reading frame, 0, 1, or 2. Combine with
                :meth:`reverse_complement` for the three reverse frames.
            to_stop: End the protein at the first stop codon, excluding the stop itself. This
                is what "the protein this ORF encodes" means; the default runs to the end of
                the sequence and marks every stop.

        Returns:
            An expression evaluating to the amino-acid sequence; null for null input.

        Raises:
            PlanError: If `frame` is not 0, 1, or 2.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ATGGCCTAAATG"]})
                >>> ds.select(p=bt.col("s").seq.translate()).to_pydict()
                {'p': ['MA*M']}

                >>> ds.select(p=bt.col("s").seq.translate(to_stop=True)).to_pydict()
                {'p': ['MA']}

                >>> # The three forward frames of a six-frame translation.
                >>> forward = [bt.col("s").seq.translate(frame=f) for f in (0, 1, 2)]
        """
        if frame not in (0, 1, 2):
            raise PlanError(f"seq.translate(): frame must be 0, 1, or 2, got {frame!r}")
        return SeqFunc("translate", self._e, frame=frame, to_stop=to_stop)

    # --- Sketching ------------------------------------------------------------------

    def kmers(self, k: int) -> SeqFunc:
        """Every length-`k` substring, sliding by one base.

        The output is a list column rather than packed integers so it composes with the
        vocabulary the engine already has: ``explode`` turns k-mers into rows for a group-by
        count, ``.list.n_unique()`` is a cardinality estimate, and ``minhash`` over the same
        list is a containment estimate.

        Upper-cased, so a soft-masked repeat counts with its unmasked copies rather than
        splitting every repeat-adjacent count in two.

        Args:
            k: The k-mer length, 1 to 256.

        Returns:
            An expression evaluating to a ``List<Utf8>``; the empty list for a sequence shorter
            than `k`, and null for null input.

        Raises:
            PlanError: If `k` is outside 1 to 256.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ACGTA"]})
                >>> ds.select(k=bt.col("s").seq.kmers(3)).to_pydict()
                {'k': [['ACG', 'CGT', 'GTA']]}

                >>> # A k-mer frequency table is an explode plus a group-by.
                >>> counts = ds.select(k=bt.col("s").seq.kmers(3)).explode("k").group_by("k")
        """
        _check_k("kmers", k)
        return SeqFunc("kmers", self._e, k=k)

    def canonical_kmers(self, k: int) -> SeqFunc:
        """K-mers folded with their reverse complements, so both strands agree.

        A sequencing read can come off either strand, so the same genomic k-mer appears as
        itself in one read and as its reverse complement in another. Canonicalization picks the
        lexicographically smaller of the two as the representative, which is what makes a
        k-mer table comparable across reads — and comparable with one built by Jellyfish, KMC,
        or minimap2, which all use the same rule.

        Args:
            k: The k-mer length, 1 to 256.

        Returns:
            An expression evaluating to a ``List<Utf8>``; the empty list for a sequence shorter
            than `k`, and null for null input.

        Raises:
            PlanError: If `k` is outside 1 to 256.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["TTT", "AAA"]})
                >>> # A read and its other-strand copy produce the same k-mer.
                >>> ds.select(k=bt.col("s").seq.canonical_kmers(3)).to_pydict()
                {'k': [['AAA'], ['AAA']]}
        """
        _check_k("canonical_kmers", k)
        return SeqFunc("canonical_kmers", self._e, k=k)

    def minimizers(self, k: int, window: int) -> SeqFunc:
        """The smallest canonical k-mer of each window, as a sketch of the sequence.

        A *minimizer* is the lexicographically smallest canonical k-mer among `window`
        consecutive k-mers. Adjacent windows overlap, so the same k-mer is usually selected
        many times running; consecutive repeats are collapsed, which is what makes the result a
        sketch — roughly ``2/(window+1)`` of the k-mers — rather than a re-encoding of the
        sequence.

        This is the primitive behind seed-and-extend alignment, and the reason two long reads
        can be compared without comparing every k-mer: two sequences sharing a substring of
        length ``window + k - 1`` are **guaranteed** to share a minimizer, so an
        ``array_intersect`` between two rows' sketches cannot miss a real overlap.

        Args:
            k: The k-mer length, 1 to 256. Alignment seeds are typically 15 to 31.
            window: How many consecutive k-mers each window spans, 1 to 256. minimap2 defaults
                to 10; a larger window is a sparser, cheaper, less sensitive sketch.

        Returns:
            An expression evaluating to a ``List<Utf8>``; null for null input.

        Raises:
            PlanError: If `k` or `window` is outside 1 to 256.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ACGTTGCAAGGCTTAACG"]})
                >>> sketch = bt.col("s").seq.minimizers(4, 5)
                >>> len(ds.select(m=sketch).to_pydict()["m"][0]) < 15
                True

                >>> # Overlap detection: how many minimizers two reads share.
                >>> shared = (
                ...     bt.col("a").seq.minimizers(15, 10)
                ...     .list.intersect(bt.col("b").seq.minimizers(15, 10))
                ...     .list.len()
                ... )
        """
        _check_k("minimizers", k)
        if not 1 <= window <= _MAX_K:
            raise PlanError(f"seq.minimizers(): window must be in 1..={_MAX_K}, got {window}")
        return SeqFunc("minimizers", self._e, k=k, window=window)

    # --- Physical properties --------------------------------------------------------

    def melting_temp(self) -> SeqFunc:
        """The duplex melting temperature in degrees Celsius.

        The quantity primer and probe design is a search over: an oligo that melts too low will
        not anneal at the annealing step, one that melts too high binds where it should not.
        Screening a candidate set becomes a filter over a computed column.

        Uses the SantaLucia (1998) unified nearest-neighbour model — the one primer3 and
        Biopython's ``Tm_NN`` default to — at 50 mM Na+ and 500 nM total strand. Those
        conditions are part of the answer: melting temperature is not a property of a sequence
        alone, and halving the concentration moves it by several degrees. A nearest-neighbour
        model is used rather than the Wallace rule or a GC-percentage formula because those
        read a sequence as a bag of bases, so ``GCGCGC`` and ``GGGCCC`` get the same answer
        despite stacking very differently.

        Returns:
            An expression evaluating to a Float64 temperature in Celsius; null for null input,
            for a sequence shorter than two bases, and for one containing any character outside
            ``ACGT`` — an ambiguity code has no defined stacking energy, and a specific
            temperature the data does not support would be worse than none.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"p": ["GTAAAACGACGGCCAGTGAA", "ACGTN"]})
                >>> tm = ds.select(tm=bt.col("p").seq.melting_temp()).to_pydict()["tm"]
                >>> 50 < tm[0] < 70, tm[1]
                (True, None)

                >>> # Keep only primers in the usable annealing range.
                >>> usable = (bt.col("p").seq.melting_temp() >= 55) & (
                ...     bt.col("p").seq.melting_temp() <= 65
                ... )
        """
        return SeqFunc("melting_temp", self._e)

    def molecular_weight(self, alphabet: str) -> SeqFunc:
        """The average molecular weight in daltons.

        For ``"protein"``, the sum of residue masses plus one water for the free termini. For
        ``"dna"`` and ``"rna"``, the sum of nucleotide monophosphate masses minus one water per
        phosphodiester bond — a **single** strand, which is what a sequence column holds;
        double it for a duplex.

        Args:
            alphabet: ``"dna"``, ``"rna"``, or ``"protein"``. The degenerate alphabets are not
                accepted: an ambiguity code has no mass, so the question has no answer.

        Returns:
            An expression evaluating to a Float64 mass in daltons; null for null input and for
            a sequence containing any character outside the alphabet's unambiguous set
            (including the ``*`` stop marker).

        Raises:
            PlanError: If `alphabet` is not ``"dna"``, ``"rna"``, or ``"protein"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ACGT"]})
                >>> mw = ds.select(mw=bt.col("s").seq.molecular_weight("dna")).to_pydict()
                >>> round(mw["mw"][0], 2)
                1253.8
        """
        _check_alphabet("molecular_weight", alphabet, _WEIGHT_ALPHABETS)
        return SeqFunc("molecular_weight", self._e, alphabet=alphabet)

    def gravy(self) -> SeqFunc:
        """The grand average of hydropathy, on the Kyte-Doolittle scale.

        The mean hydropathy index over the residues, and the standard first-pass discriminator
        between a membrane protein (typically above 0.5) and a soluble globular one (usually
        below 0).

        Residues outside the standard twenty are skipped rather than nulling the row: one ``X``
        in a long predicted protein should not erase its hydropathy.

        Returns:
            An expression evaluating to a Float64; null for null input and for a sequence with
            no scorable residue.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"p": ["IIIVVVLLL", "KKKRRRDDD"]})
                >>> g = ds.select(g=bt.col("p").seq.gravy()).to_pydict()["g"]
                >>> g[0] > 3, g[1] < -3
                (True, True)
        """
        return SeqFunc("gravy", self._e)

    def isoelectric_point(self) -> SeqFunc:
        """The pH at which the peptide carries no net charge.

        The number that decides how a protein behaves in isoelectric focusing and ion-exchange
        chromatography, and the property a purification protocol is designed around. Solved by
        bisection on the net-charge curve using the Bjellqvist pKa set, the same one ExPASy and
        Biopython use.

        Returns:
            An expression evaluating to a Float64 pH; null for null input and for the empty
            sequence.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"p": ["KKKK", "DDDD"]})
                >>> pi = ds.select(pi=bt.col("p").seq.isoelectric_point()).to_pydict()["pi"]
                >>> pi[0] > 9, pi[1] < 4.5
                (True, True)
        """
        return SeqFunc("isoelectric_point", self._e)

    # --- FASTQ quality --------------------------------------------------------------

    def phred_quality(self, *, offset: int = 33) -> SeqFunc:
        """Decode a FASTQ quality string to per-base Phred scores.

        Args:
            offset: The ASCII offset the file encodes with. 33 is Sanger and Illumina 1.8+,
                which is what every current instrument writes; 64 is the older Illumina
                1.3-1.7 pipelines. It is stated rather than sniffed because the two ranges
                overlap, so the bytes carry no reliable signal and a wrong guess shifts every
                score by 31 — turning a Q40 base into Q9. A character below the offset decodes
                to a *negative* score rather than being clamped, which is the unmistakable
                signature of the wrong choice.

        Returns:
            An expression evaluating to a ``List<Int32>``; null for null input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"q": ["!5I"]})
                >>> ds.select(p=bt.col("q").seq.phred_quality()).to_pydict()
                {'p': [[0, 20, 40]]}
        """
        return SeqFunc("phred_quality", self._e, offset=offset)

    def mean_quality(self, *, offset: int = 33) -> SeqFunc:
        """The arithmetic mean Phred score of a FASTQ quality string.

        The "average quality" every FASTQ tool reports, and what a ``mean_quality >= 20``
        filter means. It is **not** the quality corresponding to the read's average error rate:
        averaging in log space systematically overstates a read whose errors are concentrated
        in a bad tail. Use :meth:`expected_errors` when the question is how many bases are
        actually likely wrong.

        Args:
            offset: The ASCII offset, 33 or 64. See :meth:`phred_quality`.

        Returns:
            An expression evaluating to a Float64; null for null input and for an empty quality
            string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"q": ["IIII"]})
                >>> ds.select(m=bt.col("q").seq.mean_quality()).to_pydict()
                {'m': [40.0]}
        """
        return SeqFunc("mean_quality", self._e, offset=offset)

    def expected_errors(self, *, offset: int = 33) -> SeqFunc:
        """The expected number of miscalled bases in the read, ``sum(10 ** (-Q / 10))``.

        The quality filter that corresponds to an actual claim about the data: "this read
        probably contains fewer than one error" is ``expected_errors() < 1.0``. This is the
        ``fastq_maxee`` criterion from USEARCH and VSEARCH, and it is strictly more informative
        than a mean-quality threshold because it is additive over bases — one Q2 base
        contributes as much expected error as sixty Q20 bases, and a mean cannot see that.

        Args:
            offset: The ASCII offset, 33 or 64. See :meth:`phred_quality`.

        Returns:
            An expression evaluating to a Float64 (0.0 for an empty quality string); null for
            null input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"q": ["5"]})
                >>> ds.select(e=bt.col("q").seq.expected_errors()).to_pydict()
                {'e': [0.01]}

                >>> # The standard high-accuracy read filter.
                >>> clean = bt.col("q").seq.expected_errors() < 1.0
        """
        return SeqFunc("expected_errors", self._e, offset=offset)

    # --- Motif search ---------------------------------------------------------------

    def find_motif(self, motif: str) -> SeqFunc:
        """The 1-based start positions of every match of an IUPAC-degenerate motif.

        A motif is written in the degenerate alphabet: ``GGWWTT`` matches ``GGAATT``,
        ``GGATTT``, ``GGTATT``, and ``GGTTTT``. Matching is defined on *sets of bases* rather
        than on text, so ambiguity works in both directions — an ``N`` in the reference matches
        every pattern base, which a character-class regex over the literal text does not do.
        ``T`` and ``U`` are interchangeable, so an RNA motif finds a DNA site.

        Matches may **overlap**: ``AA`` occurs three times in ``AAAA``. That is the
        biologically meaningful count, and it is what distinguishes this from a
        replace-and-measure spelling, which counts only non-overlapping occurrences.

        Positions are 1-based to match every genome browser, GFF file, and VCF record you will
        compare them against.

        Args:
            motif: The pattern, in IUPAC nucleotide codes. Case-insensitive.

        Returns:
            An expression evaluating to a ``List<Int64>`` of positions; the empty list for no
            match, and null for null input.

        Raises:
            PlanError: If `motif` is empty or contains a character that is not an IUPAC
                nucleotide code — such a motif matches nowhere, and a column of empty lists
                would hide the typo.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["GGAATTCC"]})
                >>> ds.select(p=bt.col("s").seq.find_motif("GAATTC")).to_pydict()
                {'p': [[2]]}

                >>> ds.select(p=bt.col("s").seq.find_motif("GGWWTT")).to_pydict()
                {'p': [[1]]}
        """
        self._check_motif("find_motif", motif)
        return SeqFunc("find_motif", self._e, pattern=motif)

    def count_motif(self, motif: str) -> SeqFunc:
        """How many (possibly overlapping) matches of an IUPAC-degenerate motif occur.

        The reduction of :meth:`find_motif`, computed without building the position list —
        which is the difference between one integer and a list allocation per row on a
        genome-scale scan. Matching rules are identical.

        Args:
            motif: The pattern, in IUPAC nucleotide codes. Case-insensitive.

        Returns:
            An expression evaluating to an Int64 count; null for null input.

        Raises:
            PlanError: If `motif` is empty or contains a non-IUPAC character.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["AAAA", "GGAATTCC"]})
                >>> ds.select(n=bt.col("s").seq.count_motif("AA")).to_pydict()
                {'n': [3, 1]}
        """
        self._check_motif("count_motif", motif)
        return SeqFunc("count_motif", self._e, pattern=motif)

    @staticmethod
    def _check_motif(method: str, motif: str) -> None:
        """Reject a motif the engine could never match, naming the offending character.

        Checked here as well as in the engine because this is the error a caller actually hits
        — a stray ``X``, a space, or a ``-`` copied out of an alignment — and plan-build time
        is where the message can point at the position.
        """
        if not motif:
            raise PlanError(f"seq.{method}(): the motif must not be empty")
        codes = set("ACGTURYSWKMBDHVN")
        for i, ch in enumerate(motif.upper()):
            if ch not in codes:
                raise PlanError(
                    f"seq.{method}(): {ch!r} at position {i} is not an IUPAC nucleotide code"
                )
