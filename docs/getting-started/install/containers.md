# Run Batcher in containers

This page describes the prebuilt Batcher container images, how to extend them or build your own, and how to run Batcher on Docker and Kubernetes.

## What's in the images

Each release publishes two multi-architecture images to the GitHub Container Registry:

| Image | Extras installed | Use it for |
|---|---|---|
| `ghcr.io/stephenoffer/batcher:<version>` | `cloud` | Single-node jobs, including reads and writes to `s3://`, `gs://`, and `az://` |
| `ghcr.io/stephenoffer/batcher:<version>-ray` | `cloud`, `ray` | The driver, head, and worker nodes of a Ray cluster |

`latest` and `latest-ray` track the newest release. Pin a version tag in production.

:::{important}
No release has been tagged yet, so these tags aren't on the registry. Build the image from a clone instead, as {ref}`the Dockerfile section below <containers-build-from-source>` shows.
:::

Both images build on `python:3.12-slim-bookworm` and run as the non-root user `batcher` (UID 1000). They carry no CUDA libraries; for GPU work see {ref}`install-build-your-own-image`.

## Run a query with Docker

Run a query in a throwaway container:

```bash
docker run --rm ghcr.io/stephenoffer/batcher:latest \
  python -c "import batcher as bt; print(bt.from_pydict({'x': [1, 2, 3]}).agg(total=bt.col('x').sum()).to_pydict())"
```

The container prints `{'total': [6]}`.

Mount a host directory to run a script from it:

```bash
docker run --rm -v "$PWD:/work" -w /work ghcr.io/stephenoffer/batcher:latest python pipeline.py
```

Batcher sizes its threads and memory to the container's cgroup limits, so `--cpus` and `--memory` set how much it uses. Spill goes to `BATCHER_SCRATCH_DIR`; the images include a writable `/scratch` for a volume:

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

(containers-build-from-source)=
To build the engine inside the image instead of installing a published wheel, use the Dockerfile in the Batcher repository. It compiles the engine from the checkout, so it works for an unreleased revision or a network that can't reach PyPI's wheels, and it takes the extras as a build argument. From the root of a clone, run the following:

```bash
docker build -f packaging/docker/Dockerfile --build-arg EXTRAS=cloud,delta -t my-batcher .
```

## Run on Kubernetes

A Batcher job on Kubernetes is an ordinary container workload, sized to the container's limits.

:::{dropdown} Example Kubernetes Job
A Job running a pipeline script mounted from a ConfigMap, with node-local spill:

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
:::

For object-store credentials, set the environment variables {doc}`/user-guide/moving-data/cloud-storage` lists from a Kubernetes Secret, or use the node's role credentials.

To run one job across several pods, use a Ray cluster on Kubernetes through KubeRay, which {doc}`clusters-and-servers` covers.

## See also

- {doc}`clusters-and-servers`: Ray clusters, KubeRay, and air-gapped networks.
- {doc}`index`: supported platforms and the other install methods.
- {doc}`/configuration/index`: memory, spill, and parallelism settings.
