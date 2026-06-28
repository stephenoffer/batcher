import ray
ray.init(address="auto", log_to_driver=True, ignore_reinit_error=True,
         runtime_env={"py_modules":["/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/batcher/python/batcher"]})
@ray.remote(num_cpus=16)
def probe():
    import os
    a=os.cpu_count()
    try: aff=len(os.sched_getaffinity(0))
    except Exception as e: aff=str(e)
    return f"cpu_count={a} sched_affinity={aff} RAYON={os.environ.get('RAYON_NUM_THREADS')} OMP={os.environ.get('OMP_NUM_THREADS')}"
print("WORKER:", ray.get(probe.remote()), flush=True)
