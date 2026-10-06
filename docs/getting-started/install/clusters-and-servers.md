# Install on servers and clusters

This page covers installing Batcher on cloud VMs, Ray clusters including KubeRay, HPC systems, and on-premises networks with no internet access.

## Cloud VMs

A cloud VM installs Batcher exactly as a workstation does, as {doc}`python-environments` shows. pip picks the x86_64 or aarch64 wheel on its own, so Arm instances such as AWS Graviton, Google Axion, and Azure Cobalt work the same way.

## Ray clusters

Batcher runs distributed queries on Ray. Install the `ray` extra on the machine that starts the query, which Ray calls the driver:

```bash
pip install "batcher-engine[ray]"
```

### How the engine reaches the workers

You don't install Batcher on the workers yourself. When the driver connects, Batcher uploads its own package, compiled engine included, to every worker through Ray's `runtime_env`, cached by content. {doc}`/integrations/compute/ray` describes the connection options. Because the upload is the driver's build, two requirements follow:

1. Every worker must run the same operating system, processor architecture, and C library as the driver. A driver on a macOS laptop, or on an x86_64 machine attached to Arm workers, uploads an engine the workers can't load.
1. Every worker must already have Batcher's dependencies, such as PyArrow and NumPy, because the upload carries Batcher's package and not the packages it depends on. Ray itself also requires the driver and the workers to run the same Python minor version.

### Compatibility preflight

Batcher checks both requirements before any worker loads the engine. The first time a Ray session sends work to a node other than the driver's own, Batcher runs a small probe on that node. The probe uses only the Python standard library and never imports Batcher, so it can run on a node that can't load the engine. It reports the node's operating system, processor architecture, C library, Python version and implementation, and the installed versions of Batcher's dependencies, and Batcher compares the answer with the driver.

When the driver's own build is being uploaded, Batcher refuses the query with a `BackendError` if any node differs on one of the following:

- The operating system, the processor architecture, or the C library family, such as musl against glibc.
- The glibc version, when the node's glibc is older than the newest glibc symbol version the compiled engine references. A driver on a newer Linux distribution than its workers passes as long as the engine's own requirement is met.
- The Python minor version or implementation.
- PyArrow or NumPy missing from the node.

The error names each node and each field, with the node's value and the driver's. A dependency at a different version, or a missing optional dependency, is logged as a warning and doesn't stop the query.

When you set `trust_cluster_image`, or pass your own `DistributedConfig(runtime_env=...)`, the workers run a build you provided rather than the driver's. Batcher then logs the same report as a warning and runs the query, because an image built for the workers' platform is expected to differ from the driver. On a trusted image the report also names a node whose installed Batcher version differs from the driver's.

The probe runs once per node per Ray session, and nodes an autoscaler adds later are probed when they first appear. A node that doesn't answer within 60 seconds is logged as unverified and isn't refused.

### Use one image for the whole cluster

The simplest arrangement runs the driver, the head, and every worker from the same `-ray` image ({doc}`containers`), and tells Batcher to trust it instead of uploading a copy:

```bash
export BATCHER_DISTRIBUTED_TRUST_CLUSTER_IMAGE=true
```

The same setting is `DistributedConfig(trust_cluster_image=True)` in code. Set it only when every node carries the same Batcher version. A Ray job submission then runs the driver on the head node, so a laptop of any operating system can submit work.

### KubeRay

On Kubernetes, the KubeRay operator runs a Ray cluster from a `RayCluster` resource. Use the `-ray` image for the head group and every worker group. KubeRay needs the resource's `rayVersion` to match the Ray release inside the image, so read it from the image before you write the manifest:

```bash
docker run --rm ghcr.io/stephenoffer/batcher:<version>-ray python -c "import ray; print(ray.__version__)"
```

:::{dropdown} Example `RayCluster` manifest
One head and a group of workers, all on the same image, with the trust setting in every container:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: batcher
spec:
  rayVersion: "<ray version printed above>"
  headGroupSpec:
    rayStartParams: {}
    template:
      spec:
        containers:
          - name: ray-head
            image: ghcr.io/stephenoffer/batcher:<version>-ray
            env:
              - {name: BATCHER_DISTRIBUTED_TRUST_CLUSTER_IMAGE, value: "true"}
            resources:
              limits: {cpu: "4", memory: 16Gi}
  workerGroupSpecs:
    - groupName: workers
      replicas: 4
      minReplicas: 1
      maxReplicas: 16
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: ray-worker
              image: ghcr.io/stephenoffer/batcher:<version>-ray
              env:
                - {name: BATCHER_DISTRIBUTED_TRUST_CLUSTER_IMAGE, value: "true"}
              resources:
                limits: {cpu: "16", memory: 64Gi}
```
:::

Batcher inside each pod sizes itself to that container's limits. {doc}`/integrations/compute/schedulers` describes what it reads from a Kubernetes allocation.

## HPC and batch schedulers

On a Slurm, PBS, LSF, or Grid Engine system, install Batcher into a virtual environment on a filesystem that every compute node mounts, so each job step imports the same installation. Create it from a login node whose architecture matches the compute nodes:

```bash
python3 -m venv /shared/envs/batcher
/shared/envs/batcher/bin/pip install "batcher-engine[ray]"
```

Batcher reads the scheduler's grant, not the whole node, when it sizes threads, memory, and spill. {doc}`/integrations/compute/schedulers` covers each scheduler, including how to bring up Ray across a multi-node allocation.

## Air-gapped and on-premises networks

A network that can't reach PyPI installs from a private mirror, a wheelhouse, or an image archive.

::::{dropdown} A private package index
If your organization mirrors PyPI through Artifactory, Nexus, or devpi, install through the mirror:

```bash
pip install --index-url https://<your-mirror>/simple batcher-engine
```

Here `<your-mirror>` is your index's host and path. To make the mirror the default, set `PIP_INDEX_URL`, or `UV_INDEX_URL` for uv, instead of passing the flag.

::::

::::{dropdown} A wheelhouse

A *wheelhouse* is a directory holding Batcher's wheel and the wheel of every dependency. Build it on a machine with internet access, carry it in, and install from it with no index.

To build a wheelhouse for machines unlike the one you're building on, name the target platform and Python version. The following example targets Linux x86_64 and Python 3.12:

```bash
pip download "batcher-engine[cloud]" --dest wheelhouse --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --python-version 3.12
```

pip matches each `--platform` tag exactly and doesn't substitute an older, compatible one. Packages tag Linux wheels with different glibc baselines, so pass both tags shown, or pip may report that no distribution matches. For Linux on Arm, use `manylinux_2_28_aarch64` and `manylinux2014_aarch64`. For Alpine, use `musllinux_1_2_x86_64` or `musllinux_1_2_aarch64`. Leave out all three platform options when the target matches the building machine.

On the target machine, install with the index turned off:

```bash
pip install --no-index --find-links wheelhouse "batcher-engine[cloud]"
```

::::

::::{dropdown} A container image archive

If the network runs containers, save the image to a file on a connected machine and load it on the other side:

```bash
docker pull --platform linux/amd64 ghcr.io/stephenoffer/batcher:<version>
docker save ghcr.io/stephenoffer/batcher:<version> -o batcher-<version>.tar
```

```bash
docker load -i batcher-<version>.tar
```

Use `--platform linux/arm64` for Arm hosts. To place the image in an internal registry instead, tag and push it there after loading.
::::

## See also

- {doc}`containers`: the images, extending them, and Kubernetes Jobs.
- {doc}`/integrations/compute/ray`: connecting to a Ray cluster and the distributed options.
- {doc}`/integrations/compute/schedulers`: Slurm, PBS, LSF, Grid Engine, and Kubernetes allocations.
- {doc}`index`: supported platforms and the other install methods.
