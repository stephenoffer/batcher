import dataclasses, time, batcher as bt, ray
ray.init(address="auto", log_to_driver=True, ignore_reinit_error=True,
         runtime_env={"py_modules":["/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/batcher/python/batcher"],
                      "env_vars":{"AWS_DEFAULT_REGION":"us-west-2","AWS_REGION":"us-west-2","BATCHER_DBG":"1"}})
from batcher.config import active_config, set_config
set_config(active_config().replace(distributed=dataclasses.replace(active_config().distributed, ray_address="auto")))
ren={"column05":"l_extendedprice","column08":"l_returnflag","column09":"l_linestatus"}
ds=bt.read.parquet("s3://ray-benchmark-data/tpch/parquet/sf100/lineitem/*.parquet").select(*[bt.col(k).alias(v) for k,v in ren.items()])
q=lambda: ds.group_by("l_returnflag","l_linestatus").agg(rev=bt.col("l_extendedprice").sum(),n=bt.col("l_extendedprice").count()).collect(distributed=True,num_workers=8)
q()  # warm
t0=time.perf_counter(); q(); print(f"TOTAL sf100: {time.perf_counter()-t0:.1f}s", flush=True)
