#!/usr/bin/env python3
"""bench_ingest.py -- Vector ingestion benchmark for scylla-driver variants.

Measures insert throughput (rows/sec) for a given driver variant against a
ScyllaDB cluster with vector-search support.

Variants:
    A  -- scylla-driver (stock/master), BatchStatement, list[float]
    B  -- scylla-driver (enhanced), BatchStatement, numpy bulk serialization
    C  -- scylla-driver (enhanced), concurrent execute_async, numpy bulk
    D  -- python-rs-driver (Rust-backed), asyncio.gather

Usage:
    python bench_ingest.py --variant A|B|C|D \
        --host 127.0.0.1 --port 9042 \
        --dataset-dir /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m \
        --dim 768 --batch-size 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pyarrow.parquet import ParquetFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_shard_connections(session, timeout=10):
    """Wait until all shard connections are established."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pool in session._pools.values():
            if hasattr(pool.host, 'sharding_info') and pool.host.sharding_info:
                n_shards = pool.host.sharding_info.shards_count
                if len(pool._connections) >= n_shards:
                    return len(pool._connections)
        time.sleep(0.2)
    # Return what we have
    for pool in session._pools.values():
        return len(pool._connections)
    return 0


# ---------------------------------------------------------------------------
# Schema / table helpers (shared across variants)
# ---------------------------------------------------------------------------

KEYSPACE = "vdb_bench"
TABLE = "vdb_bench_collection"


def cql_create_keyspace():
    return (
        f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE} "
        f"WITH replication = {{'class': 'NetworkTopologyStrategy', "
        f"'replication_factor': '1'}} "
        f"AND tablets = {{'enabled': 'true'}}"
    )


def cql_create_table(dim: int):
    return (
        f"CREATE TABLE IF NOT EXISTS {KEYSPACE}.{TABLE} ("
        f"  id int PRIMARY KEY,"
        f"  vector vector<float, {dim}>"
        f") WITH caching = {{'keys': 'NONE', 'rows_per_partition': 'NONE'}}"
    )


CQL_TRUNCATE = f"TRUNCATE {KEYSPACE}.{TABLE}"
CQL_INSERT = f"INSERT INTO {KEYSPACE}.{TABLE} (id, vector) VALUES (?, ?)"

PROGRESS_INTERVAL = 10_000  # print progress every N rows


def _should_report(total: int, prev_reported: int) -> bool:
    """Return True when we've crossed a PROGRESS_INTERVAL boundary."""
    return total // PROGRESS_INTERVAL > prev_reported // PROGRESS_INTERVAL


# ---------------------------------------------------------------------------
# Variant A: scylla-driver (stock), BatchStatement, list[float]
# ---------------------------------------------------------------------------

def run_variant_a(host: str, port: int, dataset_dir: str, batch_size: int, dim: int, max_rows: int = 0, concurrency: int = 0):
    from cassandra import ConsistencyLevel
    from cassandra.cluster import Cluster
    from cassandra.concurrent import execute_concurrent_with_args

    conc = concurrency if concurrency > 0 else 1000
    cluster = Cluster([host], port=port, compression=False)
    session = cluster.connect()

    session.execute(cql_create_keyspace())
    session.set_keyspace(KEYSPACE)
    session.execute(cql_create_table(dim))
    session.execute(CQL_TRUNCATE)

    prepared = session.prepare(CQL_INSERT)
    prepared.consistency_level = ConsistencyLevel.ONE

    n_conns = _wait_for_shard_connections(session)
    print(f"  shard connections: {n_conns}", file=sys.stderr)

    pf = ParquetFile(f"{dataset_dir}/shuffle_train.parquet")
    total = 0
    t0 = time.perf_counter()

    for batch in pf.iter_batches(conc):
        ids = batch.column("id").to_pylist()
        embeddings = batch.column("emb").to_pylist()

        params = list(zip(ids, embeddings))
        execute_concurrent_with_args(session, prepared, params,
                                     concurrency=conc,
                                     raise_on_first_error=True)

        prev = total
        total += len(ids)
        if _should_report(total, prev):
            elapsed = time.perf_counter() - t0
            print(f"  [{total:>8,} rows] {total/elapsed:,.0f} rows/sec", file=sys.stderr)
        if max_rows > 0 and total >= max_rows:
            break

    elapsed = time.perf_counter() - t0
    cluster.shutdown()
    return total, elapsed


# ---------------------------------------------------------------------------
# Variant B: scylla-driver (enhanced), execute_concurrent, list[float] (to_pylist)
# ---------------------------------------------------------------------------

def run_variant_b(host: str, port: int, dataset_dir: str, batch_size: int, dim: int, max_rows: int = 0, concurrency: int = 0):
    from cassandra import ConsistencyLevel
    from cassandra.cluster import Cluster
    from cassandra.concurrent import execute_concurrent_with_args

    conc = concurrency if concurrency > 0 else 1000
    cluster = Cluster([host], port=port, compression=False)
    session = cluster.connect()

    session.execute(cql_create_keyspace())
    session.set_keyspace(KEYSPACE)
    session.execute(cql_create_table(dim))
    session.execute(CQL_TRUNCATE)

    prepared = session.prepare(CQL_INSERT)
    prepared.consistency_level = ConsistencyLevel.ONE

    n_conns = _wait_for_shard_connections(session)
    print(f"  shard connections: {n_conns}", file=sys.stderr)

    pf = ParquetFile(f"{dataset_dir}/shuffle_train.parquet")
    total = 0
    t0 = time.perf_counter()

    for batch in pf.iter_batches(conc):
        ids = batch.column("id").to_pylist()
        embeddings = batch.column("emb").to_pylist()

        params = list(zip(ids, embeddings))
        execute_concurrent_with_args(session, prepared, params,
                                     concurrency=conc,
                                     raise_on_first_error=True)

        prev = total
        total += len(ids)
        if _should_report(total, prev):
            elapsed = time.perf_counter() - t0
            print(f"  [{total:>8,} rows] {total/elapsed:,.0f} rows/sec", file=sys.stderr)
        if max_rows > 0 and total >= max_rows:
            break

    elapsed = time.perf_counter() - t0
    cluster.shutdown()
    return total, elapsed


# ---------------------------------------------------------------------------
# Variant C: scylla-driver (enhanced), execute_concurrent, numpy bulk
# ---------------------------------------------------------------------------

def run_variant_c(host: str, port: int, dataset_dir: str, batch_size: int, dim: int, max_rows: int = 0, concurrency: int = 0):
    import numpy as np
    from cassandra import ConsistencyLevel
    from cassandra.cluster import Cluster
    from cassandra.concurrent import execute_concurrent_with_args

    conc = concurrency if concurrency > 0 else 1000
    cluster = Cluster([host], port=port, compression=False)
    session = cluster.connect()

    session.execute(cql_create_keyspace())
    session.set_keyspace(KEYSPACE)
    session.execute(cql_create_table(dim))
    session.execute(CQL_TRUNCATE)

    prepared = session.prepare(CQL_INSERT)
    prepared.consistency_level = ConsistencyLevel.ONE

    n_conns = _wait_for_shard_connections(session)
    print(f"  shard connections: {n_conns}", file=sys.stderr)

    pf = ParquetFile(f"{dataset_dir}/shuffle_train.parquet")
    total = 0
    t0 = time.perf_counter()

    for batch in pf.iter_batches(conc):
        ids = batch.column("id").to_pylist()
        emb_col = batch.column("emb")

        # Zero-copy Arrow -> numpy, pass rows directly to driver
        arr = np.frombuffer(emb_col.values.buffers()[1], dtype=np.float32).reshape(-1, dim)

        params = [(ids[i], arr[i]) for i in range(len(ids))]
        execute_concurrent_with_args(session, prepared, params,
                                     concurrency=conc,
                                     raise_on_first_error=True)

        prev = total
        total += len(ids)
        if _should_report(total, prev):
            elapsed = time.perf_counter() - t0
            print(f"  [{total:>8,} rows] {total/elapsed:,.0f} rows/sec", file=sys.stderr)
        if max_rows > 0 and total >= max_rows:
            break

    elapsed = time.perf_counter() - t0
    cluster.shutdown()
    return total, elapsed


# ---------------------------------------------------------------------------
# Variant D: python-rs-driver, asyncio.gather
# ---------------------------------------------------------------------------

def run_variant_d(host: str, port: int, dataset_dir: str, batch_size: int, dim: int, max_rows: int = 0, concurrency: int = 0):
    import asyncio
    import numpy as np

    from scylla.enums import Consistency
    from scylla.execution_profile import ExecutionProfile
    from scylla.session_builder import SessionBuilder

    async def _run():
        profile = ExecutionProfile(consistency=Consistency.One)
        builder = SessionBuilder([host], port, execution_profile=profile)
        session = await builder.connect()

        await session.execute(cql_create_keyspace())
        await session.execute(f"USE {KEYSPACE}")
        await session.execute(cql_create_table(dim))
        await session.execute(CQL_TRUNCATE)

        prepared = await session.prepare(CQL_INSERT)
        prepared = prepared.with_consistency(Consistency.One)

        chunk_size = concurrency if concurrency > 0 else batch_size
        pf = ParquetFile(f"{dataset_dir}/shuffle_train.parquet")
        total = 0
        t0 = time.perf_counter()

        for batch in pf.iter_batches(chunk_size):
            ids = batch.column("id").to_pylist()
            emb_col = batch.column("emb")

            # Zero-copy Arrow -> numpy, then per-row tolist() for rs-driver
            arr = np.frombuffer(emb_col.values.buffers()[1], dtype=np.float32).reshape(-1, dim)

            coros = [
                session.execute(prepared, [key, arr[i].tolist()])
                for i, key in enumerate(ids)
            ]
            await asyncio.gather(*coros)

            prev = total
            total += len(ids)
            if _should_report(total, prev):
                elapsed = time.perf_counter() - t0
                print(f"  [{total:>8,} rows] {total/elapsed:,.0f} rows/sec",
                      file=sys.stderr)
            if max_rows > 0 and total >= max_rows:
                break

        elapsed = time.perf_counter() - t0
        return total, elapsed

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Variant E: scylla-driver (enhanced), decoupled executor, numpy bulk
# ---------------------------------------------------------------------------

def run_variant_e(host: str, port: int, dataset_dir: str, batch_size: int, dim: int, max_rows: int = 0, concurrency: int = 0):
    """Decoupled executor: callbacks on the event loop just signal completion,
    a dedicated submitter thread handles execute_async in batches.  This frees
    the libev event loop from doing request serialisation / routing work."""
    import numpy as np
    import threading
    from collections import deque
    from cassandra import ConsistencyLevel
    from cassandra.cluster import Cluster

    conc = concurrency if concurrency > 0 else 1000

    cluster = Cluster([host], port=port, compression=False)
    session = cluster.connect()

    session.execute(cql_create_keyspace())
    session.set_keyspace(KEYSPACE)
    session.execute(cql_create_table(dim))
    session.execute(CQL_TRUNCATE)

    prepared = session.prepare(CQL_INSERT)
    prepared.consistency_level = ConsistencyLevel.ONE

    n_conns = _wait_for_shard_connections(session)
    print(f"  shard connections: {n_conns}", file=sys.stderr)

    class _DecoupledExecutor:
        """Fire-and-forget write executor that keeps the event loop lean.

        The libev event-loop callback does only ``deque.append`` +
        ``Event.set`` (~100 ns).  A dedicated submitter thread drains
        the deque and calls ``execute_async`` in batches, which involves
        the heavier work of query-plan lookup, connection borrowing,
        message serialisation and enqueuing.
        """
        __slots__ = ('session', 'prepared', 'params', 'total', 'submitted',
                     'done_event', 'error', '_ready', '_ready_event',
                     '_stopped')

        def __init__(self, session, prepared, params, concurrency):
            self.session = session
            self.prepared = prepared
            self.params = params
            self.total = len(params)
            self.submitted = 0
            self.done_event = threading.Event()
            self.error = None
            self._ready = deque()
            self._ready_event = threading.Event()
            self._stopped = False

        def run(self):
            ea = self.session.execute_async
            prep = self.prepared
            params = self.params
            batch = min(concurrency, self.total)
            # Submit initial batch *before* starting the submitter thread
            # so self.submitted is visible without a race.
            for i in range(batch):
                f = ea(prep, params[i], timeout=None)
                f.add_callbacks(callback=self._on_done, callback_args=(f,),
                                errback=self._on_err, errback_args=(f,))
            self.submitted = batch
            # Handle edge case: fewer items than concurrency
            if self.total <= batch:
                self.done_event.wait()
                if self.error:
                    raise self.error
                return
            submitter = threading.Thread(target=self._submitter_loop,
                                        daemon=True, name="decoupled-submitter")
            submitter.start()
            self.done_event.wait()
            self._stopped = True
            self._ready_event.set()
            if self.error:
                raise self.error

        def _on_done(self, _result, future):
            future.clear_callbacks()
            self._ready.append(1)
            self._ready_event.set()

        def _on_err(self, exc, _future):
            self.error = exc
            self._stopped = True
            self.done_event.set()

        def _submitter_loop(self):
            ea = self.session.execute_async
            prep = self.prepared
            params = self.params
            total = self.total
            ready = self._ready
            ready_event = self._ready_event
            submitted = self.submitted
            completed = 0
            while not self._stopped:
                ready_event.wait()
                ready_event.clear()
                count = 0
                while True:
                    try:
                        ready.popleft()
                        count += 1
                    except IndexError:
                        break
                completed += count
                end = min(submitted + count, total)
                try:
                    for i in range(submitted, end):
                        f = ea(prep, params[i], timeout=None)
                        f.add_callbacks(callback=self._on_done, callback_args=(f,),
                                        errback=self._on_err, errback_args=(f,))
                except Exception as exc:
                    self.error = exc
                    self.done_event.set()
                    return
                submitted = end
                if completed >= total:
                    self.done_event.set()
                    return

    pf = ParquetFile(f"{dataset_dir}/shuffle_train.parquet")
    total = 0
    t0 = time.perf_counter()

    for batch in pf.iter_batches(conc):
        ids = batch.column("id").to_pylist()
        emb_col = batch.column("emb")
        arr = np.frombuffer(emb_col.values.buffers()[1], dtype=np.float32).reshape(-1, dim)
        params = [(ids[i], arr[i]) for i in range(len(ids))]

        executor = _DecoupledExecutor(session, prepared, params, conc)
        executor.run()

        prev = total
        total += len(ids)
        if _should_report(total, prev):
            elapsed = time.perf_counter() - t0
            print(f"  [{total:>8,} rows] {total/elapsed:,.0f} rows/sec", file=sys.stderr)
        if max_rows > 0 and total >= max_rows:
            break

    elapsed = time.perf_counter() - t0
    cluster.shutdown()
    return total, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = {
    "A": ("scylla-driver (master), execute_concurrent, list[float]", run_variant_a),
    "B": ("scylla-driver (enhanced), execute_concurrent, list[float]", run_variant_b),
    "C": ("scylla-driver (enhanced), execute_concurrent, numpy bulk", run_variant_c),
    "D": ("python-rs-driver, asyncio.gather", run_variant_d),
    "E": ("scylla-driver (enhanced), decoupled executor, numpy bulk", run_variant_e),
    "F": ("scylla-driver (enhanced), free-threaded, execute_concurrent, numpy bulk", run_variant_c),
}


def main():
    parser = argparse.ArgumentParser(description="Vector ingestion benchmark")
    parser.add_argument("--variant", required=True, choices=VARIANTS.keys(),
                        help="Which driver variant to benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    parser.add_argument("--dataset-dir",
                        default=os.path.expanduser("~/vector_bench/dataset/cohere/cohere_medium_1m"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Stop after this many rows (0 = all)")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Max in-flight requests for variants C/D (0 = same as batch-size)")
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    label, fn = VARIANTS[args.variant]
    concurrency = args.concurrency if args.concurrency > 0 else args.batch_size
    print(f"Running variant {args.variant}: {label}", file=sys.stderr)
    print(f"  host={args.host}:{args.port}  batch_size={args.batch_size}  dim={args.dim}  concurrency={concurrency}",
          file=sys.stderr)
    print(f"  dataset={args.dataset_dir}", file=sys.stderr)
    print("", file=sys.stderr)

    total, elapsed = fn(args.host, args.port, args.dataset_dir, args.batch_size, args.dim, args.max_rows, concurrency)

    rows_per_sec = total / elapsed if elapsed > 0 else 0
    result = {
        "variant": args.variant,
        "label": label,
        "rows": total,
        "elapsed_sec": round(elapsed, 2),
        "rows_per_sec": round(rows_per_sec, 1),
    }

    # Print JSON to stdout (machine-readable), human-readable to stderr
    print(json.dumps(result))
    print("", file=sys.stderr)
    print(f"  Result: {total:,} rows in {elapsed:.1f}s = {rows_per_sec:,.0f} rows/sec",
          file=sys.stderr)


if __name__ == "__main__":
    main()
