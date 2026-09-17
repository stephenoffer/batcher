# Run Batcher in containers

This page describes the prebuilt Batcher container images, how to extend them or build your own, and how to run Batcher on Docker and Kubernetes.

## What's in the images

Each release publishes two images to the GitHub Container Registry. Both are multi-architecture, so the same tag pulls the right build on an x86_64 host and on an Arm host such as AWS Graviton or an Apple silicon Mac:

| Image | Extras installed | Use it for |
|---|---|---|
| `ghcr.io/stephenoffer/batcher:<version>` | `cloud` | Single-node jobs, including reads and writes to `s3://`, `gs://`, and `az://` |
| `ghcr.io/stephenoffer/batcher:<version>-ray` | `cloud`, `ray` | The driver, head, and worker nodes of a Ray cluster |

`latest` and `latest-ray` track the newest release. Pin a version tag in production, so a new release can't change a running job underneath you.

Both images start from the official `python:3.12-slim-bookworm` image, install the same wheel that PyPI serves, and run as a non-root user named `batcher` with UID 1000. The default command is `python`. The images carry no CUDA libraries, so GPU work needs its own image, as {ref}`install-build-your-own-image` describes.

## Run a query with Docker

To check that the image runs on your host, start a container and run a query in it:

```bash
docker run --rm ghcr.io/stephenoffer/batcher:latest \
  python -c "import batcher as bt; print(bt.from_pydict({'x': [1, 2, 3]}).agg(total=bt.col('x').sum()).to_pydict())"
```

The container prints `{'total': [6]}`.

To work on files from the host, mount a directory and run a script from it. The `batcher` user must be able to read the mount, and to write to it if the job writes output there:

```bash
docker run --rm -v "$PWD:/work" -w /work ghcr.io/stephenoffer/batcher:latest python pipeline.py
```

Batcher reads the container's CPU quota and memory limit from its cgroup. It sizes its thread count to the quota and measures memory pressure against the limit, rather than against the whole host, so `--cpus` and `--memory` change how much Batcher uses.

When a query spills to disk, Batcher writes to the directory in `BATCHER_SCRATCH_DIR`. Without it, Batcher picks the fastest local volume it can measure, and falls back to the system temporary directory, which in a container is the writable layer. The images include an empty `/scratch` directory owned by the `batcher` user, so a named volume mounted there starts out writable. Mount one for spill and name it:

```bash
docker run --rm -v "$PWD:/work" -w /work -v batcher-spill:/scratch -e BATCHER_SCRATCH_DIR=/scratch \
  ghcr.io/stephenoffer/batcher:latest python pipeline.py
```

## Extend a prebuilt image

To add extras or your own code, start from a prebuilt image. The images run as a non-root user, so switch to `root` to install packages and switch back afterward:

```dockerfile
FROM ghcr.io/stephenoffer/batcher:<version>

USER root
RUN pip install --no-cache-dir "batcher-engine[delta,kafka]"
USER batcher

COPY pipeline.py /home/batcher/
CMD ["python", "pipeline.py"]
```

pip treats the engine already in the image as satisfying `batcher-engine`, so it installs the packages the extras need without changing the engine version.

(install-build-your-own-image)=
## Build your own image

Any image that has Python 3.11 or newer on a supported platform can install Batcher with pip, so a base image you already maintain works as well as the prebuilt ones:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "batcher-engine[cloud]"
```

Alpine and other musl-based images install the musllinux wheel the same way, with no compiler in the image, because PyArrow and NumPy publish musllinux wheels too:

```dockerfile
FROM python:3.12-alpine
RUN pip install --no-cache-dir batcher-engine
```

For GPU work, start from a CUDA base image that carries the Python version you need, then install the extras for your framework, such as `torch` or `vllm`. {doc}`/ml/index` describes what each ML extra requires.

To build the engine inside the image instead of installing a published wheel, use the Dockerfile in the Batcher repository. It compiles the engine from the checkout, so it works for an unreleased revision or a network that can't reach PyPI's wheels, and it takes the extras as a build argument. From the root of a clone, run the following:

```bash
docker build -f packaging/docker/Dockerfile --build-arg EXTRAS=cloud,delta -t my-batcher .
```

## Run on Kubernetes

A Batcher job on Kubernetes is an ordinary container workload. Set memory and CPU limits on the container, because those limits are what Batcher sizes itself to. The following Job runs a pipeline script mounted from a ConfigMap:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nightly-etl
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      securityContext:
        runAsUser: 1000
        runAsNonRoot: true
      containers:
        - name: batcher
          image: ghcr.io/stephenoffer/batcher:<version>
          command: ["python", "/pipeline/pipeline.py"]
          resources:
            requests: {cpu: "8", memory: 32Gi}
            limits: {cpu: "8", memory: 32Gi}
          env:
            - {name: BATCHER_SCRATCH_DIR, value: /scratch}
          volumeMounts:
            - {name: pipeline, mountPath: /pipeline}
            - {name: spill, mountPath: /scratch}
      volumes:
        - name: pipeline
          configMap: {name: nightly-etl-pipeline}
        - name: spill
          emptyDir: {}
```

The `emptyDir` mounted at `/scratch`, named by `BATCHER_SCRATCH_DIR`, gives spill files node-local disk rather than the container's writable layer. For object-store credentials, set the environment variables {doc}`/user-guide/moving-data/cloud-storage` lists, from a Kubernetes Secret, or rely on the node's instance or role credentials.

To run one job across several pods, use a Ray cluster on Kubernetes through KubeRay, which {doc}`clusters-and-servers` covers.

## See also

- {doc}`clusters-and-servers`: Ray clusters, KubeRay, and air-gapped networks.
- {doc}`index`: supported platforms and the other install methods.
- {doc}`/configuration/index`: memory, spill, and parallelism settings.
