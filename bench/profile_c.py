#!/usr/bin/env python3
"""Profile variant C (enhanced driver + numpy) to find driver bottlenecks.

Usage:
  # Phase-level timing breakdown:
  taskset -c 4-5 python profile_c.py

  # cProfile of the full run:
  taskset -c 4-5 python profile_c.py --cprofile

  # py-spy flamegraph (run separately):
  taskset -c 4-5 py-spy record -o flame.svg -- python profile_c.py
"""

import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np
from pyarrow.parquet import ParquetFile

from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args

KEYSPACE = "vdb_bench"
TABLE = "vdb_bench_collection"
DIM = 768
CONCURRENCY = 1000

CQL_CREATE_KS = (
    f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE} "
    f"WITH replication = {{'class': 'NetworkTopologyStrategy', 'replication_factor': '1'}} "
    f"AND tablets = {{'enabled': 'true'}}"
)
CQL_CREATE_TABLE = (
    f"CREATE TABLE IF NOT EXISTS {KEYSPACE}.{TABLE} ("
    f"  id int PRIMARY KEY, vector vector<float, {DIM}>"
    f") WITH caching = {{'keys': 'NONE', 'rows_per_partition': 'NONE'}}"
)
CQL_TRUNCATE = f"TRUNCATE {KEYSPACE}.{TABLE}"
CQL_INSERT = f"INSERT INTO {KEYSPACE}.{TABLE} (id, vector) VALUES (?, ?)"


def _wait_for_shard_connections(session, timeout=10):
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        for pool in session._pools.values():
            if len(pool._connections) >= 4:
                return len(pool._connections)
        _time.sleep(0.5)
    for pool in session._pools.values():
        return len(pool._connections)
    return 0


def setup_session(host, port):
    cluster = Cluster([host], port=port, compression=False)
    session = cluster.connect()
    session.execute(CQL_CREATE_KS)
    session.set_keyspace(KEYSPACE)
    session.execute(CQL_CREATE_TABLE)
    session.execute(CQL_TRUNCATE)
    prepared = session.prepare(CQL_INSERT)
    prepared.consistency_level = ConsistencyLevel.ONE
    n = _wait_for_shard_connections(session)
    print(f"  shard connections: {n}", file=sys.stderr)
    return cluster, session, prepared


def run_phased_profile(host, port, dataset_dir, max_rows):
    """Detailed per-phase timing breakdown."""
    cluster, session, prepared = setup_session(host, port)
    pf = ParquetFile(f"{dataset_dir}/shuffle_train.parquet")

    # Accumulators
    t_parquet = 0.0
    t_numpy = 0.0
    t_zip = 0.0
    t_execute = 0.0
    total = 0

    t0 = time.perf_counter()

    for batch in pf.iter_batches(CONCURRENCY):
        # Phase 1: column extraction + id to_pylist
        t1 = time.perf_counter()
        ids = batch.column("id").to_pylist()
        emb_col = batch.column("emb")
        t2 = time.perf_counter()
        t_parquet += t2 - t1

        # Phase 2: zero-copy Arrow -> numpy
        arr = np.frombuffer(emb_col.values.buffers()[1], dtype=np.float32).reshape(-1, DIM)
        t3 = time.perf_counter()
        t_numpy += t3 - t2

        # Phase 3: build params list
        params = [(ids[i], arr[i]) for i in range(len(ids))]
        t4 = time.perf_counter()
        t_zip += t4 - t3

        # Phase 4: execute_concurrent (serialize + dispatch + wait)
        execute_concurrent_with_args(session, prepared, params,
                                     concurrency=CONCURRENCY,
                                     raise_on_first_error=True)
        t5 = time.perf_counter()
        t_execute += t5 - t4

        total += len(ids)
        if total >= max_rows:
            break

    elapsed = time.perf_counter() - t0

    print(f"\n{'='*65}", file=sys.stderr)
    print(f" Phase Profile: {total:,} rows in {elapsed:.1f}s = {total/elapsed:,.0f} rows/sec", file=sys.stderr)
    print(f"{'='*65}", file=sys.stderr)
    print(f"  {'Phase':<30} {'Time (s)':>8} {'%':>6} {'per row (us)':>12}", file=sys.stderr)
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*12}", file=sys.stderr)

    phases = [
        ("parquet col + id to_pylist", t_parquet),
        ("np.frombuffer (zero-copy)", t_numpy),
        ("build params list", t_zip),
        ("execute_concurrent (all)", t_execute),
    ]

    accounted = sum(t for _, t in phases)
    overhead = elapsed - accounted

    for name, t in phases:
        pct = 100 * t / elapsed
        per_row = 1e6 * t / total
        print(f"  {name:<30} {t:>8.2f} {pct:>5.1f}% {per_row:>10.1f}us", file=sys.stderr)

    pct = 100 * overhead / elapsed
    per_row = 1e6 * overhead / total
    print(f"  {'(iter/overhead)':<30} {overhead:>8.2f} {pct:>5.1f}% {per_row:>10.1f}us", file=sys.stderr)
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*12}", file=sys.stderr)
    print(f"  {'TOTAL':<30} {elapsed:>8.2f} 100.0%  {1e6*elapsed/total:>10.1f}us", file=sys.stderr)

    cluster.shutdown()


def run_cprofile(host, port, dataset_dir, max_rows):
    """cProfile the full run to see driver-internal hotspots."""
    cluster, session, prepared = setup_session(host, port)
    pf = ParquetFile(f"{dataset_dir}/shuffle_train.parquet")

    def _ingest():
        total = 0
        for batch in pf.iter_batches(CONCURRENCY):
            ids = batch.column("id").to_pylist()
            emb_col = batch.column("emb")
            arr = np.frombuffer(emb_col.values.buffers()[1], dtype=np.float32).reshape(-1, DIM)
            params = [(ids[i], arr[i]) for i in range(len(ids))]
            execute_concurrent_with_args(session, prepared, params,
                                         concurrency=CONCURRENCY,
                                         raise_on_first_error=True)
            total += len(ids)
            if total >= max_rows:
                break
        return total

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    total = _ingest()
    profiler.disable()
    elapsed = time.perf_counter() - t0

    print(f"\n  {total:,} rows in {elapsed:.1f}s = {total/elapsed:,.0f} rows/sec\n", file=sys.stderr)

    stats = pstats.Stats(profiler, stream=sys.stderr)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    print("\n=== Top 40 by cumulative time ===", file=sys.stderr)
    stats.print_stats(40)

    stats.sort_stats('tottime')
    print("\n=== Top 40 by total (self) time ===", file=sys.stderr)
    stats.print_stats(40)

    cluster.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    parser.add_argument("--dataset-dir",
                        default=os.path.expanduser("~/vector_bench/dataset/cohere/cohere_medium_1m"))
    parser.add_argument("--cprofile", action="store_true",
                        help="Run with cProfile instead of phase timing")
    parser.add_argument("--max-rows", type=int, default=200_000)
    args = parser.parse_args()

    if args.cprofile:
        run_cprofile(args.host, args.port, args.dataset_dir, args.max_rows)
    else:
        run_phased_profile(args.host, args.port, args.dataset_dir, args.max_rows)


if __name__ == "__main__":
    main()
