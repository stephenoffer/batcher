"""What to *do* about a fault, as distinct from what the fault was.

`xid` classifies a code as hardware, application, or unknown. That is the question a scheduler
asks, and it is not the question anyone else asks. Two more follow from it, and each one has a
different answer and a different audience:

* **What repairs the device?** A device that needs a reset comes back in seconds and belongs in
  a drain list. One that needs a power cycle needs a human or an out-of-band controller. One
  whose spare memory rows are exhausted needs an RMA and will never come back, and leaving it
  in a "retry after reset" loop means a slot that is permanently down while looking temporary.
* **Can the results already produced on it be trusted?** This is the one nobody asks and the
  one that costs the most when it is wrong. A double-bit ECC error means a tensor read back is
  not the tensor that was written — the arithmetic that followed it completed successfully and
  returned a wrong number. A job that survives that fault by retrying the *next* task has kept
  a corrupted answer and will write it out with every appearance of success.

Both are derived from the same published Xid table `xid` reads, and both refuse to guess: an
unrecognized code is `"investigate"` and is *not* assumed to have preserved data integrity in
either direction. Where the two disagree the caller wants both, which is why they are separate
functions rather than one verdict.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.hardware.faults.xid import XID_APPLICATION, XID_FATAL, describe_xid

__all__ = [
    "REMEDY_ORDER",
    "XID_REMEDY",
    "XID_UNTRUSTED",
    "device_remedy",
    "explain_codes",
    "undocumented_remedies",
    "xid_remedy",
    "xid_untrusted",
]

#: Remedies in increasing order of how much they cost to apply. `device_remedy` reports the
#: most expensive one implied by a set of codes, because a device that needs both a reset and
#: an RMA needs the RMA — acting on the cheaper answer produces a repair loop that never
#: converges and a slot that is down while its ticket says "pending reset".
REMEDY_ORDER = ("none", "fix_application", "reset", "power_cycle", "replace")

#: The repair each documented code calls for, from the remedy column of NVIDIA's own Xid table.
#:
#: Only codes this module is confident about appear. An absent code reports `"investigate"`,
#: which is the honest answer and also the safe one: it tells an operator to read the vendor's
#: documentation rather than handing them an invented remedy that will not work.
XID_REMEDY: dict[int, str] = {
    13: "fix_application",  # illegal memory access in a kernel — the board is fine
    31: "fix_application",  # MMU fault from an out-of-bounds access
    43: "fix_application",  # stopped after a fault elsewhere in the same process
    45: "fix_application",  # preemptive cleanup after a process was killed
    48: "reset",  # double-bit ECC
    62: "reset",  # micro-controller halt
    63: "reset",  # row remap recorded — takes effect at the next reset
    64: "replace",  # row remap *failed*: the spare pool is gone and no reset restores it
    68: "reset",  # video processor exception
    74: "reset",  # NVLink error; the fabric retrains on reset
    79: "power_cycle",  # fallen off the bus — the driver cannot re-enumerate it in place
    92: "reset",  # high single-bit ECC rate
    94: "reset",  # contained ECC error, the faulting process is dead
    95: "reset",  # uncontained ECC error, every process on the device is suspect
    119: "reset",  # GSP RPC timeout
    120: "reset",  # GSP error
    121: "reset",  # C2C corrected error rate high
}

#: Codes after which data already produced on the device must be treated as wrong.
#:
#: The distinction from "fatal" is the whole point and it cuts both ways. Xid 79 is as fatal as
#: a code gets — the board is off the bus — and it corrupts nothing, because a device that
#: vanished returns no results at all; the tasks on it fail loudly and are retried. Xid 48 and
#: 95 are the opposite and are the dangerous half: the device kept running, kept answering, and
#: the answers are wrong. A job that retries past one of these completes successfully with
#: corrupt output, which is strictly worse than the crash it avoided.
XID_UNTRUSTED = frozenset(
    {
        48,  # double-bit ECC: the read returned data that is not what was written
        95,  # uncontained ECC: the fault escaped its process, so any resident tensor is suspect
    }
)


def xid_remedy(code: int) -> str:
    """What repairs a device that reported this code.

    Args:
        code: The Xid number.

    Returns:
        One of `REMEDY_ORDER`, or `"investigate"` for a code outside the documented set. Never
        a guess — a future driver release must not be able to send a fleet through a repair
        cycle this build invented for a code it has never seen.
    """
    return XID_REMEDY.get(code, "investigate")


def xid_untrusted(code: int) -> bool:
    """Whether results computed on the device before this fault may be wrong.

    The question that decides whether a job may retry past a fault or must fail. A device that
    took a double-bit ECC error returned a number, and the number is wrong; retrying the *next*
    task keeps the wrong one and writes it out looking like a success.

    Args:
        code: The Xid number.

    Returns:
        True only for the codes documented as returning corrupted data. False for every other
        code, including unrecognized ones — an unknown code is not evidence of corruption, and
        treating it as such would fail every job on a fleet the day a driver adds a code.
    """
    return code in XID_UNTRUSTED


def device_remedy(codes: tuple[int, ...]) -> str:
    """The repair a device needs, given every code reported against it.

    Args:
        codes: Xid numbers seen for one device.

    Returns:
        The most expensive remedy any code implies, from `REMEDY_ORDER`; `"investigate"` when
        a code is unrecognized and nothing more expensive was seen; `"none"` for no codes.
    """
    remedies = [xid_remedy(code) for code in codes]
    known = [r for r in remedies if r in REMEDY_ORDER]
    if not known:
        return "investigate" if remedies else "none"
    worst = max(known, key=REMEDY_ORDER.index)
    # An unrecognized code alongside a merely-application one is not settled by the application
    # answer. A device that reported a workload bug and also something this build has never
    # seen must not be reported as needing nothing but a code fix.
    if len(known) < len(remedies) and REMEDY_ORDER.index(worst) <= REMEDY_ORDER.index("reset"):
        return "investigate"
    return worst


def explain_codes(codes: tuple[int, ...]) -> str:
    """One line naming what each code means and what it calls for.

    The string an operator reads out of a quarantine log. Kept here rather than built at each
    call site so a fleet's incident reports say the same thing about the same code.

    Args:
        codes: Xid numbers seen for one device.

    Returns:
        A comma-separated description, `""` for no codes.
    """
    return ", ".join(f"Xid {code} ({describe_xid(code)})" for code in codes)


def undocumented_remedies() -> tuple[int, ...]:
    """Classified Xid codes this module has no repair for.

    The remedy table is derived from the severity table in `xid` and must not drift away from
    it: a code that `xid` calls fatal but that has no repair listed here quarantines a device
    and then gives an operator nothing to do about it, which is exactly the failure mode where
    a fleet shrinks and no ticket explains it. Exposed as a function rather than enforced by a
    module-level assertion because assertions vanish under `python -O`, and a consistency check
    that stops running in production is worse than none.

    Returns:
        The codes `xid` classifies but `XID_REMEDY` does not cover, ascending. Empty when the
        two tables agree, which is the state a unit test holds them in.
    """
    classified = XID_FATAL | XID_APPLICATION.keys()
    return tuple(sorted(classified - XID_REMEDY.keys()))
