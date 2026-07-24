#!/usr/bin/env bash
# Rebuild the Rust dev toolchain after the machine underneath the workspace is replaced.
#
# WHY THIS EXISTS
#
# On an Anyscale workspace only the workspace directory is snapshotted and restored. `$HOME`
# is not: it comes back from the container image. A default `rustup` install puts the
# toolchain in `~/.cargo` and `~/.rustup`, so **every node restart silently destroys the
# ability to build this repo** — `cargo`, `rustc` and `just` are simply gone, and `just
# check` fails with "command not found" rather than anything that names the cause. That is
# exactly what a restart on 2026-07-24 did.
#
# WHERE THINGS GO, AND WHY NOT SOMEWHERE SIMPLER
#
#   toolchain + registry -> $BATCHER_TOOLS_DIR   (local NVMe: fast, lost on restart)
#   restore tarball      -> $BATCHER_TOOLS_CACHE (cluster storage: slow, survives restart)
#
# Putting the toolchain straight onto cluster storage is the obvious one-line fix and it is
# the wrong one. That mount is NFS, and while its bulk throughput is fine (~137 MB/s here)
# its metadata latency is ~11 ms per file create against ~0.035 ms on local disk — a ~300x
# penalty on precisely the many-small-files access pattern a cargo registry and a `target/`
# directory are made of. Measured: `cargo check` on this workspace had not finished
# downloading the registry index after four minutes from NFS. So the toolchain runs from
# local disk and cluster storage holds a *tarball* of it, which is a bulk read and plays to
# what NFS is good at. Restore costs one sequential read instead of a network install.
#
# USAGE
#
#   source tools/bootstrap_env.sh     # export the env vars; restore from cache if needed
#   bash   tools/bootstrap_env.sh     # same, then report what it did
#
# Sourced from a shell startup file it is a no-op once the toolchain is present, so the cost
# on every later shell is a few `test -x` calls.

_bt_repo_root() {
    # `BASH_SOURCE`/`$0` both work when sourced from bash or zsh; fall back to the cwd so a
    # `curl | bash` style invocation still lands somewhere sane.
    local self="${BASH_SOURCE[0]:-${(%):-%x}}"
    if [ -n "$self" ] && [ -e "$self" ]; then
        (cd "$(dirname "$self")/.." && pwd -P)
    else
        pwd -P
    fi
}

BATCHER_TOOLS_DIR="${BATCHER_TOOLS_DIR:-/mnt/local_storage/batcher-tools}"
BATCHER_TOOLS_CACHE="${BATCHER_TOOLS_CACHE:-/mnt/cluster_storage/batcher-toolchain}"
_bt_tarball="$BATCHER_TOOLS_CACHE/toolchain.tar.zst"

export CARGO_HOME="$BATCHER_TOOLS_DIR/cargo"
export RUSTUP_HOME="$BATCHER_TOOLS_DIR/rustup"
# `target/` is a rebuildable cache and the single largest small-file workload in the repo, so
# it belongs on local disk too. Keeping it out of the workspace directory also keeps it out
# of the 5-minute autosnapshot, which would otherwise zip several GB of object files.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BATCHER_TOOLS_DIR/target}"

case ":$PATH:" in
    *":$CARGO_HOME/bin:"*) ;;
    *) export PATH="$CARGO_HOME/bin:$PATH" ;;
esac

_bt_have_toolchain() { [ -x "$CARGO_HOME/bin/cargo" ] && [ -x "$CARGO_HOME/bin/just" ]; }

# Snapshot the working toolchain so the next restart restores instead of re-downloading.
_bt_save_cache() {
    mkdir -p "$BATCHER_TOOLS_CACHE" || return 1
    local tmp="$_bt_tarball.$$"
    # `--exclude` the registry sources: they are ~5x the toolchain and cargo re-extracts them
    # from the (cached) .crate files on demand.
    if tar -C "$BATCHER_TOOLS_DIR" \
        --exclude='cargo/registry/src' --exclude='target' \
        -I 'zstd -T0 -3' -cf "$tmp" cargo rustup 2>/dev/null; then
        mv -f "$tmp" "$_bt_tarball"
        echo "batcher: cached toolchain -> $_bt_tarball"
    else
        rm -f "$tmp"
        return 1
    fi
}

_bt_restore_cache() {
    [ -f "$_bt_tarball" ] || return 1
    mkdir -p "$BATCHER_TOOLS_DIR" || return 1
    echo "batcher: restoring Rust toolchain from $_bt_tarball ..." >&2
    tar -C "$BATCHER_TOOLS_DIR" -I 'zstd -d' -xf "$_bt_tarball" 2>/dev/null && _bt_have_toolchain
}

_bt_install_from_network() {
    local root; root="$(_bt_repo_root)"
    echo "batcher: installing Rust toolchain to $BATCHER_TOOLS_DIR (no cache) ..." >&2
    mkdir -p "$CARGO_HOME" "$RUSTUP_HOME" || return 1
    # rust-toolchain.toml pins the channel and components, so let rustup read it rather than
    # restating the pin here and letting the two drift.
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal || return 1
    (cd "$root" && "$CARGO_HOME/bin/rustup" show >/dev/null) || return 1

    # `just` is a single static binary; take the release rather than `cargo install` (which
    # would compile it, for minutes, every restore).
    local url
    url=$(curl -sSL https://api.github.com/repos/casey/just/releases/latest \
        | grep -o '"browser_download_url": *"[^"]*x86_64-unknown-linux-musl[^"]*"' \
        | head -1 | cut -d'"' -f4)
    [ -n "$url" ] || return 1
    curl -sSfL "$url" | tar -xz -C "$CARGO_HOME/bin" just || return 1
    _bt_have_toolchain
}

bt_bootstrap() {
    if _bt_have_toolchain; then
        return 0
    fi
    if _bt_restore_cache; then
        echo "batcher: toolchain restored." >&2
        return 0
    fi
    if _bt_install_from_network; then
        _bt_save_cache
        return 0
    fi
    echo "batcher: could not provision the Rust toolchain; run 'bash tools/bootstrap_env.sh'" >&2
    return 1
}

bt_bootstrap

# Run (not sourced) => report the resulting environment, so the script doubles as a check.
if [ "${BASH_SOURCE[0]:-}" = "${0:-}" ]; then
    echo "CARGO_HOME=$CARGO_HOME"
    echo "RUSTUP_HOME=$RUSTUP_HOME"
    echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
    "$CARGO_HOME/bin/cargo" --version 2>/dev/null || echo "cargo: MISSING"
    "$CARGO_HOME/bin/just" --version 2>/dev/null || echo "just: MISSING"
fi
