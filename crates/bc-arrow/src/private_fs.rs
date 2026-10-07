//! Creating the engine's on-disk artifacts owner-only, in one place.
//!
//! Every file the data plane writes to a scratch path holds the query's **actual rows**: a
//! spilled partition, a shuffle bucket published through `/dev/shm`, a gathered reducer
//! input staged out of core. On a Ray worker those paths are node-local directories other
//! tenants also mount, and at the default umask they land 0755/0644 — readable by any local
//! user, and in the shared-memory case *writable*, so a planted file that decodes cleanly is
//! read as authoritative shuffle data and silently changes the answer.
//!
//! The mode is set in the `open` call rather than by a following `chmod`, because a chmod
//! leaves a window in which the file is world-readable and a reader that wins that race gets
//! everything.
//!
//! This lives in `bc-arrow` for the same reason `_internal/paths.py` lives in a neutral leaf
//! on the control-plane side: `bc-runtime` (spill), `bc-transport` (the shm fast path) and
//! `bc-py` (the gather staging files) all write artifacts, and `bc-arrow` is the lowest crate
//! all three can see. Each of them had grown its own copy — `create_private`,
//! `create_private_file`, and, in `bc-py`, a plain `File::create` that had no copy at all and
//! wrote the gathered bucket 0644.

use std::fs::File;
use std::path::Path;

/// Create (or truncate) `path` for writing, owner-only from the moment it exists.
///
/// On a non-unix platform this is `File::create`: there is no mode to set, and refusing to
/// write would be worse than the exposure the platform does not have.
///
/// # Errors
/// Whatever `OpenOptions::open` returns — a missing parent, a permission denial, a full disk.
pub fn create_private_file(path: &Path) -> std::io::Result<File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

/// Create `dir` and its parents owner-only, tightening it if it already exists.
///
/// `DirBuilder::mode` honours the process umask, so it *requests* 0700 and may get less, and
/// a recursive create silently leaves an existing directory's mode alone — which is the
/// common case for a scratch root an earlier run created 0755. So the mode is both requested
/// and, on unix, asserted afterwards.
///
/// The assertion is best-effort: a directory this process does not own (a mount an operator
/// set up) cannot be tightened, and failing the query over that would be worse than the
/// exposure, because the files written inside are created 0600 regardless.
///
/// # Errors
/// Whatever `DirBuilder::create` returns. A failure to *tighten* an existing directory is
/// not an error.
pub fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)?;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
        Ok(())
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

/// Whether `dir` is a real directory this process owns and no one else can enter.
///
/// [`create_private_dir`] cannot promise that. It tolerates a chmod it could not apply and
/// follows a symlink, and both are what a local attacker reaches for on a shared,
/// world-writable mount: pre-create the path 0777 under their own uid, and the "private"
/// directory is one they can rename a planted file into; make it a symlink, and a later
/// `remove_dir_all` beneath it deletes under whatever it points at. A caller about to *trust*
/// what it reads back from the directory -- the same-node shm fast path, which maps a bucket
/// as authoritative shuffle data -- asks this afterwards and uses the directory only on
/// `true`. Checked with `symlink_metadata`, so a link is judged as a link, not as its target.
///
/// On a non-unix platform there is no owner or mode to check, so any real directory passes,
/// matching what `create_private_dir` can do there.
#[must_use]
pub fn verify_private_dir(dir: &Path) -> bool {
    let Ok(meta) = std::fs::symlink_metadata(dir) else {
        return false;
    };
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.uid() == effective_uid() && meta.mode() & 0o077 == 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(unix)]
fn effective_uid() -> u32 {
    // SAFETY: `geteuid` takes no arguments, touches no memory, and cannot fail.
    unsafe { libc::geteuid() }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    /// A unique scratch root for one test, removed on the way out. The workspace carries no
    /// `tempfile` dependency and this crate is the DAG root, so adding one here would put it
    /// under every crate in the build.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("bc-private-fs-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("scratch root");
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn a_created_file_is_owner_only() {
        let scratch = Scratch::new("file");
        let path = scratch.join("rows.arrow");
        create_private_file(&path).unwrap();
        assert_eq!(mode(&path), 0o600, "artifact is readable beyond the owner");
    }

    #[test]
    fn a_created_directory_is_owner_only() {
        let scratch = Scratch::new("dir");
        let nested = scratch.join("a").join("b");
        create_private_dir(&nested).unwrap();
        assert_eq!(mode(&nested), 0o700);
    }

    /// The common real case: a scratch root some earlier run left 0755. A recursive create
    /// succeeds and leaves the mode alone, so without the explicit tightening the very first
    /// deployment to reuse a scratch path would be unprotected — and nothing would say so.
    #[test]
    fn an_existing_world_readable_directory_is_tightened() {
        let scratch = Scratch::new("existing");
        let existing = scratch.join("reused");
        std::fs::create_dir(&existing).unwrap();
        std::fs::set_permissions(&existing, std::fs::Permissions::from_mode(0o755)).unwrap();
        create_private_dir(&existing).unwrap();
        assert_eq!(mode(&existing), 0o700);
    }

    /// Truncation, because a bucket path is reused across a re-split and a short second write
    /// must not leave the first write's tail behind it.
    #[test]
    fn creating_over_an_existing_file_truncates_it() {
        use std::io::Write;
        let scratch = Scratch::new("truncate");
        let path = scratch.join("bucket.arrow");
        create_private_file(&path)
            .unwrap()
            .write_all(b"0123456789")
            .unwrap();
        create_private_file(&path)
            .unwrap()
            .write_all(b"ab")
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"ab");
    }

    #[test]
    fn a_created_directory_verifies_as_private() {
        let scratch = Scratch::new("verify-ok");
        let dir = scratch.join("d");
        create_private_dir(&dir).unwrap();
        assert!(verify_private_dir(&dir));
    }

    /// The mode a chmod could not tighten -- a directory another uid planted -- is refused,
    /// not accepted because `create_private_dir` returned `Ok`.
    #[test]
    fn a_group_or_world_accessible_directory_does_not_verify() {
        let scratch = Scratch::new("verify-mode");
        let dir = scratch.join("loose");
        std::fs::create_dir(&dir).unwrap();
        for mode in [0o777, 0o750, 0o701] {
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
            assert!(
                !verify_private_dir(&dir),
                "mode {mode:o} verified as private"
            );
        }
    }

    /// A symlink to a private directory is still a symlink: whoever could create the link
    /// decides where reads and deletes beneath it land.
    #[test]
    fn a_symlink_to_a_private_directory_does_not_verify() {
        let scratch = Scratch::new("verify-link");
        let target = scratch.join("target");
        create_private_dir(&target).unwrap();
        let link = scratch.join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(verify_private_dir(&target));
        assert!(!verify_private_dir(&link));
    }

    /// A directory another uid owns does not verify even at 0700. `/root` is the one such
    /// directory a test box reliably has; the case is vacuous when it is absent or when the
    /// test runs as root, so it asserts only when the precondition holds.
    #[test]
    fn a_directory_owned_by_another_user_does_not_verify() {
        use std::os::unix::fs::MetadataExt;
        let other = Path::new("/root");
        if let Ok(m) = std::fs::symlink_metadata(other) {
            if m.is_dir() && m.uid() != effective_uid() && m.mode() & 0o077 == 0 {
                assert!(!verify_private_dir(other));
            }
        }
    }

    #[test]
    fn a_missing_path_or_a_file_does_not_verify() {
        let scratch = Scratch::new("verify-missing");
        assert!(!verify_private_dir(&scratch.join("absent")));
        let file = scratch.join("f");
        create_private_file(&file).unwrap();
        assert!(!verify_private_dir(&file));
    }
}
