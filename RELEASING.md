# Releasing

Batcher is published to PyPI as **`batcher-engine`** and imported as **`batcher`**.
Releases are built and published by `.github/workflows/release.yml`: it builds
[abi3](https://docs.python.org/3/c-api/stable.html) wheels (one per platform covers
Python 3.11+) for Linux glibc and musl on x86_64 and aarch64, macOS on x86_64 and arm64,
and Windows x86_64, plus an sdist. It installs every wheel on its own platform and runs
`packaging/smoke_test.py`, and on a `v*` tag publishes the wheels to PyPI and multi-arch
images to `ghcr.io/stephenoffer/batcher` (`:<version>`, `:<version>-ray`, `:latest`,
`:latest-ray`) from `packaging/docker/Dockerfile`.

The platform list is every platform pyarrow publishes wheels for. Adding one pyarrow does not
cover (Windows on ARM, ppc64le, s390x) produces a wheel no one can install.

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

## One-time container registry setup

Nothing to configure before the first release: the `docker` job pushes with the workflow's
own `GITHUB_TOKEN`. GitHub creates a new container package as **private**, though, so after
the first tagged release:

1. Open `https://github.com/users/stephenoffer/packages/container/package/batcher`.
2. **Package settings** → **Change visibility** → **Public**.

Until it is public, `docker pull ghcr.io/stephenoffer/batcher` fails for everyone but you.

## Cutting a release

1. Bump the version in the workspace `Cargo.toml` (`[workspace.package].version`).
   The Python package version is derived from it (`dynamic = ["version"]`).
2. Commit the bump.
3. Tag and push — the tag must match the version:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The workflow builds all wheels + sdist and smoke-tests them, then the `publish` job (gated
on the tag) uploads them to PyPI, and the `docker` job pushes the images after that succeeds.
Pushes and pull requests run the build, smoke, and image-build jobs as a check; they never
publish.

## conda-forge (once, after the first PyPI release)

conda-forge builds from the PyPI sdist and is reviewed by conda-forge maintainers, so it can't
be published from this repository. `packaging/conda-forge/recipe.yaml` is ready to submit; it
was built and its tests passed locally with `rattler-build` against conda-forge's compilers.

1. Get the sdist's hash:

   ```bash
   curl -sL https://pypi.org/packages/source/b/batcher-engine/batcher_engine-0.1.0.tar.gz | sha256sum
   ```

2. In `recipe.yaml`, set `version` to the released version and `sha256` to that hash.
3. Fork `conda-forge/staged-recipes`, copy the recipe to `recipes/batcher-engine/recipe.yaml`,
   and open a pull request. Every GitHub handle under `recipe-maintainers` must comment on it
   to accept.
4. When it merges, conda-forge creates `conda-forge/batcher-engine-feedstock`, and its bot
   opens a version-bump pull request for every later PyPI release.
5. Update the "conda, mamba, and pixi" section of
   `docs/getting-started/install/python-environments.md`, which tells readers Batcher is not on
   conda-forge and to install from PyPI instead.

## Installing without a release

To install an unreleased revision straight from source (builds the Rust engine, so a
[Rust toolchain](https://rustup.rs) is required):

```bash
pip install "git+https://github.com/stephenoffer/batcher.git"
```
