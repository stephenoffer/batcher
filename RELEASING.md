# Releasing

Batcher is published to PyPI as **`batcher-engine`** and imported as **`batcher`**.
Releases are built and published by `.github/workflows/release.yml`: it builds
[abi3](https://docs.python.org/3/c-api/stable.html) wheels (one per platform covers
Python 3.10+) for Linux, macOS, and Windows plus an sdist, and publishes them to PyPI
when a `v*` tag is pushed.

## One-time PyPI setup (Trusted Publishing)

The workflow publishes via [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — there is **no API token stored in the repo**. Configure it once:

1. Create the project on PyPI (a maintainer must reserve the `batcher-engine` name —
   either by an initial manual `maturin publish`, or via the
   [pending-publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   flow which lets the first trusted-publisher run create the project).
2. On PyPI → `batcher-engine` → **Publishing** → add a GitHub Actions publisher:
   - Owner: `stephenoffer`
   - Repository: `batcher`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In the GitHub repo → Settings → Environments, create an environment named `pypi`
   (optionally add required reviewers to gate publishes).

## Cutting a release

1. Bump the version in the workspace `Cargo.toml` (`[workspace.package].version`).
   The Python package version is derived from it (`dynamic = ["version"]`).
2. Commit the bump.
3. Tag and push — the tag must match the version:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The workflow builds all wheels + sdist, then the `publish` job (gated on the tag)
uploads them to PyPI. Pushes and pull requests run the build jobs only as a sanity
check; they never publish.

## Installing without a release

To install an unreleased revision straight from source (builds the Rust engine, so a
[Rust toolchain](https://rustup.rs) is required):

```bash
pip install "git+https://github.com/stephenoffer/batcher.git"
```
