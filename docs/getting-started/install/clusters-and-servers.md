# Install on servers and clusters

This page covers installing Batcher on cloud VMs, Ray clusters including KubeRay, HPC systems, and on-premises networks with no internet access.

## Cloud VMs

A cloud VM installs Batcher exactly as a workstation does, with pip or uv into a virtual environment, as {doc}`python-environments` shows. The only choice the cloud adds is the processor architecture, and both kinds of instance have a wheel. x86_64 instances use the x86_64 wheel. Arm instances, such as AWS Graviton, Google Axion, and Azure Cobalt, use the aarch64 wheel, and pip picks the right one on its own.

The x86_64 wheel requires the `x86-64-v2` instruction set, which every x86_64 instance type provides, and it uses AVX2 or AVX-512 on the instances that have them. {doc}`index` has the details.

## Ray clusters

Batcher runs distributed queries on Ray, so a cluster needs the `ray` extra on the machine that starts the query, which Ray calls the driver:

```bash
pip install "batcher-engine[ray]"
```

### How the engine reaches the workers

By default you don't install Batcher on the worker nodes yourself. When the driver connects to a cluster, Batcher uploads its own installed package, compiled engine included, to every worker through Ray's `runtime_env`. Ray caches the upload by its contents, so a cluster receives each build of Batcher once. {doc}`/integrations/compute/ray` describes the connection options.

The upload is the driver's own build of the engine, and two requirements follow from that:

1. Every worker must run the same operating system, processor architecture, and C library as the driver. A driver on a macOS laptop, or on an x86_64 machine attached to Arm workers, uploads an engine the workers can't load.
1. Every worker must already have Batcher's dependencies, such as PyArrow and NumPy, because the upload carries Batcher's package and not the packages it depends on. Ray itself also requires the driver and the workers to run the same Python minor version.

### Use one image for the whole cluster

The arrangement that satisfies both requirements everywhere is to run the driver, the head, and every worker from the same image, and tell Batcher to trust it rather than upload its own copy. The `-ray` image, described in {doc}`containers`, carries Batcher, its dependencies, and Ray for exactly this purpose.

Set the following environment variable wherever the driver runs, such as in the image or the job's environment:

```bash
export BATCHER_DISTRIBUTED_TRUST_CLUSTER_IMAGE=true
```

With it set, the driver skips the upload and uses the engine each node already has. The same setting is available in code as `DistributedConfig(trust_cluster_image=True)`. Only set it when every node really does carry the same Batcher version, because a worker with a different engine from the driver fails at the first task.

A Ray job submission runs the driver on the head node rather than on the machine that submits it, so with this arrangement a laptop of any operating system or architecture can submit work to the cluster.

### KubeRay

On Kubernetes, the KubeRay operator runs a Ray cluster from a `RayCluster` resource. Use the `-ray` image for the head group and every worker group. KubeRay needs the resource's `rayVersion` to match the Ray release inside the image, so read it from the image before you write the manifest:

```bash
docker run --rm ghcr.io/stephenoffer/batcher:<version>-ray python -c "import ray; print(ray.__version__)"
```

The following `RayCluster` runs one head and a group of workers, all on the same image, with the trust setting in every container's environment:

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

Batcher inside each pod sizes itself to that container's limits. {doc}`/integrations/compute/schedulers` describes what it reads from a Kubernetes allocation.

## HPC and batch schedulers

On a Slurm, PBS, LSF, or Grid Engine system, install Batcher into a virtual environment on a filesystem that every compute node mounts, so each job step imports the same installation. Create it from a login node whose architecture matches the compute nodes:

```bash
python3 -m venv /shared/envs/batcher
/shared/envs/batcher/bin/pip install "batcher-engine[ray]"
```

Batcher reads the scheduler's grant, not the whole node, when it sizes threads, memory, and spill. {doc}`/integrations/compute/schedulers` covers each scheduler, including how to bring up Ray across a multi-node allocation.

## Air-gapped and on-premises networks

A network that can't reach PyPI installs from files carried in. There are three ways to do that, depending on what the network already runs.

### A private package index

If your organization mirrors PyPI through a proxy such as Artifactory, Nexus, or devpi, install through the mirror:

```bash
pip install --index-url https://<your-mirror>/simple batcher-engine
```

Here `<your-mirror>` is your index's host and path. To make the mirror the default, set `PIP_INDEX_URL`, or `UV_INDEX_URL` for uv, instead of passing the flag.

### A wheelhouse

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

### A container image archive

If the network runs containers, save the image to a file on a connected machine and load it on the other side:

```bash
docker pull --platform linux/amd64 ghcr.io/stephenoffer/batcher:<version>
docker save ghcr.io/stephenoffer/batcher:<version> -o batcher-<version>.tar
```

```bash
docker load -i batcher-<version>.tar
```

Use `--platform linux/arm64` for Arm hosts. To place the image in an internal registry instead, tag and push it there after loading.

## See also

- {doc}`containers`: the images, extending them, and Kubernetes Jobs.
- {doc}`/integrations/compute/ray`: connecting to a Ray cluster and the distributed options.
- {doc}`/integrations/compute/schedulers`: Slurm, PBS, LSF, Grid Engine, and Kubernetes allocations.
- {doc}`index`: supported platforms and the other install methods.
