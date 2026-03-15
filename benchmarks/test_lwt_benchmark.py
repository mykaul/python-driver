# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmarks for LWT (Lightweight Transaction) prepared statement performance.

These benchmarks measure the client-side overhead of the LWT prepared
statement pipeline without requiring a live cluster.  They cover:

1. **Bind benchmarks** (test_bind_*):  BoundStatement.bind() with schemas
   typical of LWT queries (INSERT IF NOT EXISTS, UPDATE IF condition).

2. **Decode benchmarks** (test_decode_*):  Result row deserialization for
   both LWT outcomes -- applied (1 column) and not-applied (N columns with
   existing values).

3. **Row factory benchmarks** (test_row_factory_*):  named_tuple_factory and
   tuple_factory overhead for LWT-shaped result sets.

4. **was_applied benchmarks** (test_was_applied_*):  ResultSet.was_applied
   property overhead for different statement types.

5. **Comparison benchmarks** (test_compare_*):  Head-to-head comparison of
   prepared LWT vs prepared non-LWT (plain INSERT/SELECT) to quantify the
   LWT-specific overhead.

Run with:
    pytest benchmarks/test_lwt_benchmark.py -v
    pytest benchmarks/test_lwt_benchmark.py -v --benchmark-sort=name
    pytest benchmarks/test_lwt_benchmark.py -v --benchmark-group-by=group
"""

import struct
from collections import namedtuple
from unittest.mock import Mock

import pytest

from cassandra import cqltypes
from cassandra.protocol import ColumnMetadata
from cassandra.query import (
    BoundStatement,
    PreparedStatement,
    named_tuple_factory,
    tuple_factory,
    dict_factory,
    BatchStatement,
    SimpleStatement,
)
from cassandra.cluster import ResultSet


# ---------------------------------------------------------------------------
# Helpers: PreparedStatement construction
# ---------------------------------------------------------------------------


def _make_lwt_insert_prepared(ncols, protocol_version=4):
    """
    Create a PreparedStatement simulating INSERT ... IF NOT EXISTS.

    Bind columns: primary key + value columns (all Int32Type).
    result_metadata is None (LWT result schema varies).
    """
    column_metadata = [
        ColumnMetadata("ks", "tbl", "pk", cqltypes.Int32Type),
    ] + [
        ColumnMetadata("ks", "tbl", "col_%d" % i, cqltypes.Int32Type)
        for i in range(ncols - 1)
    ]
    return PreparedStatement(
        column_metadata=column_metadata,
        query_id=b"lwt-insert-id",
        routing_key_indexes=[0],
        query="INSERT INTO ks.tbl (pk, %s) VALUES (?, %s) IF NOT EXISTS"
        % (
            ", ".join("col_%d" % i for i in range(ncols - 1)),
            ", ".join("?" for _ in range(ncols - 1)),
        ),
        keyspace="ks",
        protocol_version=protocol_version,
        result_metadata=None,  # LWT: variable result schema
        result_metadata_id=b"meta-id-lwt",
        is_lwt=True,
    )


def _make_lwt_update_prepared(ncols, protocol_version=4):
    """
    Create a PreparedStatement simulating UPDATE ... IF condition.

    Bind columns: value columns + primary key + condition value (all Int32Type).
    """
    # SET col_0 = ?, col_1 = ?, ... WHERE pk = ? IF col_0 = ?
    column_metadata = [
        ColumnMetadata("ks", "tbl", "col_%d" % i, cqltypes.Int32Type)
        for i in range(ncols - 2)
    ] + [
        ColumnMetadata("ks", "tbl", "pk", cqltypes.Int32Type),
        ColumnMetadata("ks", "tbl", "cond_col", cqltypes.Int32Type),
    ]
    return PreparedStatement(
        column_metadata=column_metadata,
        query_id=b"lwt-update-id",
        routing_key_indexes=[ncols - 2],
        query="UPDATE ks.tbl SET %s WHERE pk = ? IF cond_col = ?"
        % ", ".join("col_%d = ?" % i for i in range(ncols - 2)),
        keyspace="ks",
        protocol_version=protocol_version,
        result_metadata=None,
        result_metadata_id=b"meta-id-lwt-upd",
        is_lwt=True,
    )


def _make_plain_insert_prepared(ncols, protocol_version=4):
    """
    Create a PreparedStatement for a plain INSERT (non-LWT) for comparison.

    result_metadata is set (server metadata can be cached).
    """
    column_metadata = [
        ColumnMetadata("ks", "tbl", "pk", cqltypes.Int32Type),
    ] + [
        ColumnMetadata("ks", "tbl", "col_%d" % i, cqltypes.Int32Type)
        for i in range(ncols - 1)
    ]
    result_metadata = [("ks", "tbl", "[applied]", cqltypes.BooleanType)]
    return PreparedStatement(
        column_metadata=column_metadata,
        query_id=b"plain-insert-id",
        routing_key_indexes=[0],
        query="INSERT INTO ks.tbl (pk, %s) VALUES (?, %s)"
        % (
            ", ".join("col_%d" % i for i in range(ncols - 1)),
            ", ".join("?" for _ in range(ncols - 1)),
        ),
        keyspace="ks",
        protocol_version=protocol_version,
        result_metadata=result_metadata,
        result_metadata_id=b"meta-id-plain",
        is_lwt=False,
    )


def _make_plain_select_prepared(ncols, protocol_version=4):
    """
    Create a PreparedStatement for a plain SELECT (non-LWT) for comparison.
    """
    column_metadata = [
        ColumnMetadata("ks", "tbl", "pk", cqltypes.Int32Type),
    ]
    result_metadata = [
        ("ks", "tbl", "pk", cqltypes.Int32Type),
    ] + [("ks", "tbl", "col_%d" % i, cqltypes.Int32Type) for i in range(ncols - 1)]
    return PreparedStatement(
        column_metadata=column_metadata,
        query_id=b"plain-select-id",
        routing_key_indexes=[0],
        query="SELECT * FROM ks.tbl WHERE pk = ?",
        keyspace="ks",
        protocol_version=protocol_version,
        result_metadata=result_metadata,
        result_metadata_id=b"meta-id-sel",
        is_lwt=False,
    )


# ---------------------------------------------------------------------------
# Helpers: Value generation
# ---------------------------------------------------------------------------


def _int32_values(n):
    """Generate a tuple of n int32 values for binding."""
    return tuple(range(1, n + 1))


# ---------------------------------------------------------------------------
# Helpers: LWT result simulation
# ---------------------------------------------------------------------------


def _lwt_applied_row():
    """Simulate LWT result when applied: ([applied],) = (True,)"""
    return [(True,)]


def _lwt_applied_colnames():
    return ("[applied]",)


def _lwt_not_applied_row(ncols):
    """
    Simulate LWT result when NOT applied:
    ([applied], pk, col_0, ..., col_N) = (False, existing_values...)
    """
    return [(False,) + tuple(range(ncols))]


def _lwt_not_applied_colnames(ncols):
    return ("[applied]", "pk") + tuple("col_%d" % i for i in range(ncols - 1))


def _plain_select_rows(nrows, ncols):
    """Simulate plain SELECT result rows."""
    return [tuple(range(ncols)) for _ in range(nrows)]


def _plain_select_colnames(ncols):
    return ("pk",) + tuple("col_%d" % i for i in range(ncols - 1))


# ---------------------------------------------------------------------------
# Helpers: Mock ResponseFuture for ResultSet
# ---------------------------------------------------------------------------


def _make_response_future(row_factory, query=None, col_names=None, col_types=None):
    """Create a minimal Mock ResponseFuture for ResultSet construction."""
    mock = Mock()
    mock.row_factory = row_factory
    mock.has_more_pages = False
    mock._col_names = col_names
    mock._col_types = col_types
    mock.query = query
    return mock


# ============================================================================
# 1. BIND BENCHMARKS
# ============================================================================


class TestBindLwtInsert:
    """Benchmark BoundStatement.bind() for LWT INSERT IF NOT EXISTS."""

    @pytest.mark.benchmark(group="bind-lwt-insert")
    @pytest.mark.parametrize("ncols", [3, 5, 10, 20])
    def test_bind_lwt_insert(self, benchmark, ncols):
        prepared = _make_lwt_insert_prepared(ncols)
        values = _int32_values(ncols)

        def do_bind():
            bs = BoundStatement(prepared)
            bs.bind(values)
            return bs

        result = benchmark(do_bind)
        assert len(result.values) == ncols


class TestBindLwtUpdate:
    """Benchmark BoundStatement.bind() for LWT UPDATE ... IF condition."""

    @pytest.mark.benchmark(group="bind-lwt-update")
    @pytest.mark.parametrize("ncols", [4, 6, 10, 20])
    def test_bind_lwt_update(self, benchmark, ncols):
        prepared = _make_lwt_update_prepared(ncols)
        values = _int32_values(ncols)

        def do_bind():
            bs = BoundStatement(prepared)
            bs.bind(values)
            return bs

        result = benchmark(do_bind)
        assert len(result.values) == ncols


class TestBindDictLwt:
    """Benchmark dict-based binding for LWT prepared statements."""

    @pytest.mark.benchmark(group="bind-lwt-dict")
    @pytest.mark.parametrize("ncols", [3, 5, 10])
    def test_bind_lwt_insert_dict(self, benchmark, ncols):
        prepared = _make_lwt_insert_prepared(ncols)
        col_names = [cm.name for cm in prepared.column_metadata]
        values_dict = {name: i + 1 for i, name in enumerate(col_names)}

        def do_bind():
            bs = BoundStatement(prepared)
            bs.bind(values_dict)
            return bs

        result = benchmark(do_bind)
        assert len(result.values) == ncols


class TestBindComparison:
    """Compare LWT vs non-LWT bind performance (should be identical)."""

    @pytest.mark.benchmark(group="bind-compare")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_bind_lwt(self, benchmark, ncols):
        prepared = _make_lwt_insert_prepared(ncols)
        values = _int32_values(ncols)

        def do_bind():
            bs = BoundStatement(prepared)
            bs.bind(values)

        benchmark(do_bind)

    @pytest.mark.benchmark(group="bind-compare")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_bind_plain(self, benchmark, ncols):
        prepared = _make_plain_insert_prepared(ncols)
        values = _int32_values(ncols)

        def do_bind():
            bs = BoundStatement(prepared)
            bs.bind(values)

        benchmark(do_bind)


# ============================================================================
# 2. DECODE BENCHMARKS (Python-level row deserialization)
# ============================================================================


def _build_int32_binary_rows(nrows, ncols):
    """
    Build a list of rows in the format that recv_results_rows produces
    before applying from_binary: list of tuples of raw bytes.

    For Int32Type, each value is a 4-byte big-endian int.
    """
    rows = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            row.append(struct.pack(">i", r * ncols + c))
        rows.append(row)
    return rows


def _decode_rows_python(raw_rows, coltypes, protocol_version=4):
    """
    Simulate the Python decode path: call from_binary for each cell.
    This mirrors recv_results_rows in protocol.py (non-Cython path).
    """
    decoded = []
    for row in raw_rows:
        decoded_row = tuple(
            coltype.from_binary(val, protocol_version)
            for val, coltype in zip(row, coltypes)
        )
        decoded.append(decoded_row)
    return decoded


class TestDecodeLwtApplied:
    """Benchmark decoding LWT 'applied' results (1 row, 1 column: [applied]=True)."""

    @pytest.mark.benchmark(group="decode-lwt")
    def test_decode_lwt_applied(self, benchmark):
        # [applied] is BooleanType: 1 byte
        raw_rows = [
            [b"\x01"],  # True
        ]
        coltypes = [cqltypes.BooleanType]

        result = benchmark(_decode_rows_python, raw_rows, coltypes)
        assert result == [(True,)]


class TestDecodeLwtNotApplied:
    """
    Benchmark decoding LWT 'not applied' results.
    Returns [applied]=False plus existing column values.
    """

    @pytest.mark.benchmark(group="decode-lwt")
    @pytest.mark.parametrize("ncols", [3, 5, 10, 20])
    def test_decode_lwt_not_applied(self, benchmark, ncols):
        # [applied] (bool) + pk (int32) + (ncols-2) value columns (int32)
        raw_rows = [
            [b"\x00"]  # applied=False
            + [struct.pack(">i", i) for i in range(ncols - 1)]
        ]
        coltypes = [cqltypes.BooleanType] + [cqltypes.Int32Type] * (ncols - 1)

        result = benchmark(_decode_rows_python, raw_rows, coltypes)
        assert result[0][0] is False
        assert len(result[0]) == ncols


class TestDecodePlainSelect:
    """Benchmark decoding plain SELECT results for comparison."""

    @pytest.mark.benchmark(group="decode-compare")
    @pytest.mark.parametrize(
        "nrows,ncols",
        [(1, 5), (1, 10), (10, 5), (10, 10), (100, 5)],
    )
    def test_decode_plain_select(self, benchmark, nrows, ncols):
        raw_rows = _build_int32_binary_rows(nrows, ncols)
        coltypes = [cqltypes.Int32Type] * ncols

        result = benchmark(_decode_rows_python, raw_rows, coltypes)
        assert len(result) == nrows
        assert len(result[0]) == ncols


class TestDecodeLwtVsPlain:
    """
    Compare per-row decode cost: LWT not-applied (1 row) vs plain SELECT (1 row).
    Both have the same number of columns -- the difference is the BooleanType
    first column in LWT vs Int32Type in SELECT.
    """

    @pytest.mark.benchmark(group="decode-1row-compare")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_decode_1row_lwt_not_applied(self, benchmark, ncols):
        raw_rows = [[b"\x00"] + [struct.pack(">i", i) for i in range(ncols - 1)]]
        coltypes = [cqltypes.BooleanType] + [cqltypes.Int32Type] * (ncols - 1)
        benchmark(_decode_rows_python, raw_rows, coltypes)

    @pytest.mark.benchmark(group="decode-1row-compare")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_decode_1row_plain_select(self, benchmark, ncols):
        raw_rows = _build_int32_binary_rows(1, ncols)
        coltypes = [cqltypes.Int32Type] * ncols
        benchmark(_decode_rows_python, raw_rows, coltypes)


# ============================================================================
# 3. ROW FACTORY BENCHMARKS
# ============================================================================


class TestRowFactoryLwt:
    """Benchmark row factory overhead for LWT-shaped results."""

    @pytest.mark.benchmark(group="row-factory-lwt")
    def test_named_tuple_factory_lwt_applied(self, benchmark):
        colnames = _lwt_applied_colnames()
        rows = _lwt_applied_row()
        benchmark(named_tuple_factory, colnames, rows)

    @pytest.mark.benchmark(group="row-factory-lwt")
    @pytest.mark.parametrize("ncols", [5, 10, 20])
    def test_named_tuple_factory_lwt_not_applied(self, benchmark, ncols):
        colnames = _lwt_not_applied_colnames(ncols)
        rows = _lwt_not_applied_row(ncols)
        benchmark(named_tuple_factory, colnames, rows)

    @pytest.mark.benchmark(group="row-factory-lwt")
    def test_tuple_factory_lwt_applied(self, benchmark):
        colnames = _lwt_applied_colnames()
        rows = _lwt_applied_row()
        benchmark(tuple_factory, colnames, rows)

    @pytest.mark.benchmark(group="row-factory-lwt")
    @pytest.mark.parametrize("ncols", [5, 10, 20])
    def test_tuple_factory_lwt_not_applied(self, benchmark, ncols):
        colnames = _lwt_not_applied_colnames(ncols)
        rows = _lwt_not_applied_row(ncols)
        benchmark(tuple_factory, colnames, rows)

    @pytest.mark.benchmark(group="row-factory-lwt")
    def test_dict_factory_lwt_applied(self, benchmark):
        colnames = _lwt_applied_colnames()
        rows = _lwt_applied_row()
        benchmark(dict_factory, colnames, rows)

    @pytest.mark.benchmark(group="row-factory-lwt")
    @pytest.mark.parametrize("ncols", [5, 10, 20])
    def test_dict_factory_lwt_not_applied(self, benchmark, ncols):
        colnames = _lwt_not_applied_colnames(ncols)
        rows = _lwt_not_applied_row(ncols)
        benchmark(dict_factory, colnames, rows)


class TestRowFactoryComparison:
    """Compare row factory cost: LWT 1-row vs plain multi-row."""

    @pytest.mark.benchmark(group="row-factory-compare")
    def test_named_tuple_lwt_1row_5col(self, benchmark):
        colnames = _lwt_not_applied_colnames(5)
        rows = _lwt_not_applied_row(5)
        benchmark(named_tuple_factory, colnames, rows)

    @pytest.mark.benchmark(group="row-factory-compare")
    def test_named_tuple_plain_1row_5col(self, benchmark):
        colnames = _plain_select_colnames(5)
        rows = _plain_select_rows(1, 5)
        benchmark(named_tuple_factory, colnames, rows)

    @pytest.mark.benchmark(group="row-factory-compare")
    def test_named_tuple_plain_100row_5col(self, benchmark):
        colnames = _plain_select_colnames(5)
        rows = _plain_select_rows(100, 5)
        benchmark(named_tuple_factory, colnames, rows)


# ============================================================================
# 4. was_applied BENCHMARKS
# ============================================================================


class TestWasApplied:
    """Benchmark ResultSet.was_applied for different statement types."""

    @pytest.mark.benchmark(group="was-applied")
    def test_was_applied_named_tuple_applied(self, benchmark):
        Row = namedtuple("Row", ["applied"])
        query = Mock()
        query.is_lwt.return_value = True
        rf = _make_response_future(
            named_tuple_factory,
            query=query,
            col_names=["[applied]"],
        )
        rs = ResultSet(rf, [Row(applied=True)])

        result = benchmark(lambda: rs.was_applied)
        assert result is True

    @pytest.mark.benchmark(group="was-applied")
    def test_was_applied_named_tuple_not_applied(self, benchmark):
        Row = namedtuple("Row", ["applied", "pk", "col_0"])
        query = Mock()
        query.is_lwt.return_value = True
        rf = _make_response_future(
            named_tuple_factory,
            query=query,
            col_names=["[applied]", "pk", "col_0"],
        )
        rs = ResultSet(rf, [Row(applied=False, pk=1, col_0=42)])

        result = benchmark(lambda: rs.was_applied)
        assert result is False

    @pytest.mark.benchmark(group="was-applied")
    def test_was_applied_tuple_factory(self, benchmark):
        query = Mock()
        query.is_lwt.return_value = True
        rf = _make_response_future(
            tuple_factory,
            query=query,
            col_names=["[applied]"],
        )
        rs = ResultSet(rf, [(True,)])

        result = benchmark(lambda: rs.was_applied)
        assert result is True

    @pytest.mark.benchmark(group="was-applied")
    def test_was_applied_dict_factory(self, benchmark):
        query = Mock()
        query.is_lwt.return_value = True
        rf = _make_response_future(
            dict_factory,
            query=query,
            col_names=["[applied]"],
        )
        rs = ResultSet(rf, [{"[applied]": True}])

        result = benchmark(lambda: rs.was_applied)
        assert result is True

    @pytest.mark.benchmark(group="was-applied")
    def test_was_applied_batch_statement(self, benchmark):
        """Benchmark was_applied when query is a BatchStatement (triggers regex check)."""
        batch = BatchStatement()
        rf = _make_response_future(
            tuple_factory,
            query=batch,
            col_names=["[applied]"],
        )
        rs = ResultSet(rf, [(True,)])

        result = benchmark(lambda: rs.was_applied)
        assert result is True

    @pytest.mark.benchmark(group="was-applied")
    def test_was_applied_simple_batch_string(self, benchmark):
        """Benchmark was_applied with a SimpleStatement that looks like a batch (regex path)."""
        ss = SimpleStatement("BEGIN BATCH INSERT INTO tbl (pk) VALUES (1) APPLY BATCH")
        rf = _make_response_future(
            tuple_factory,
            query=ss,
            col_names=["[applied]"],
        )
        rs = ResultSet(rf, [(True,)])

        result = benchmark(lambda: rs.was_applied)
        assert result is True


# ============================================================================
# 5. FULL PIPELINE COMPARISON (bind + decode + row_factory)
# ============================================================================


class TestFullPipelineComparison:
    """
    Simulate the full client-side pipeline: bind() then decode+row_factory.
    This measures combined overhead without network I/O.
    """

    @pytest.mark.benchmark(group="full-pipeline")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_pipeline_lwt_insert_applied(self, benchmark, ncols):
        """LWT INSERT IF NOT EXISTS, applied=True result."""
        prepared = _make_lwt_insert_prepared(ncols)
        values = _int32_values(ncols)
        raw_result = [b"\x01"]  # applied=True
        coltypes = [cqltypes.BooleanType]
        colnames = ("[applied]",)

        def pipeline():
            bs = BoundStatement(prepared)
            bs.bind(values)
            decoded = _decode_rows_python([raw_result], coltypes)
            return named_tuple_factory(colnames, decoded)

        result = benchmark(pipeline)
        assert len(result) == 1

    @pytest.mark.benchmark(group="full-pipeline")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_pipeline_lwt_insert_not_applied(self, benchmark, ncols):
        """LWT INSERT IF NOT EXISTS, not applied -- returns existing values."""
        prepared = _make_lwt_insert_prepared(ncols)
        values = _int32_values(ncols)
        raw_result = [b"\x00"] + [struct.pack(">i", i) for i in range(ncols)]
        coltypes = [cqltypes.BooleanType] + [cqltypes.Int32Type] * ncols
        colnames = ("[applied]", "pk") + tuple("col_%d" % i for i in range(ncols - 1))

        def pipeline():
            bs = BoundStatement(prepared)
            bs.bind(values)
            decoded = _decode_rows_python([raw_result], coltypes)
            return named_tuple_factory(colnames, decoded)

        result = benchmark(pipeline)
        assert len(result) == 1

    @pytest.mark.benchmark(group="full-pipeline")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_pipeline_plain_insert(self, benchmark, ncols):
        """Plain INSERT (non-LWT) for comparison."""
        prepared = _make_plain_insert_prepared(ncols)
        values = _int32_values(ncols)

        def pipeline():
            bs = BoundStatement(prepared)
            bs.bind(values)
            # Plain INSERT returns RESULT_KIND_VOID, no rows to decode
            return None

        benchmark(pipeline)

    @pytest.mark.benchmark(group="full-pipeline")
    @pytest.mark.parametrize("ncols", [5, 10])
    def test_pipeline_plain_select_1row(self, benchmark, ncols):
        """Plain SELECT returning 1 row for comparison with LWT."""
        prepared = _make_plain_select_prepared(ncols)
        values = _int32_values(1)  # WHERE pk = ?
        raw_rows = _build_int32_binary_rows(1, ncols)
        coltypes = [cqltypes.Int32Type] * ncols
        colnames = _plain_select_colnames(ncols)

        def pipeline():
            bs = BoundStatement(prepared)
            bs.bind(values)
            decoded = _decode_rows_python(raw_rows, coltypes)
            return named_tuple_factory(colnames, decoded)

        result = benchmark(pipeline)
        assert len(result) == 1


# ============================================================================
# 6. LWT-SPECIFIC PROPERTIES
# ============================================================================


class TestLwtProperties:
    """Benchmark is_lwt() checks and routing_key computation for LWT."""

    @pytest.mark.benchmark(group="lwt-properties")
    def test_is_lwt_prepared(self, benchmark):
        prepared = _make_lwt_insert_prepared(5)
        benchmark(prepared.is_lwt)

    @pytest.mark.benchmark(group="lwt-properties")
    def test_is_lwt_bound(self, benchmark):
        prepared = _make_lwt_insert_prepared(5)
        bound = prepared.bind(_int32_values(5))
        benchmark(bound.is_lwt)

    @pytest.mark.benchmark(group="lwt-properties")
    def test_is_lwt_batch(self, benchmark):
        batch = BatchStatement()
        prepared = _make_lwt_insert_prepared(3)
        batch.add(prepared, _int32_values(3))

        benchmark(batch.is_lwt)

    @pytest.mark.benchmark(group="lwt-properties")
    def test_routing_key_lwt_bound(self, benchmark):
        prepared = _make_lwt_insert_prepared(5)
        bound = prepared.bind(_int32_values(5))

        # First access computes and caches
        _ = bound.routing_key

        benchmark(lambda: bound.routing_key)

    @pytest.mark.benchmark(group="lwt-properties")
    def test_skip_meta_lwt_vs_plain(self, benchmark):
        """
        Measure the bool(result_metadata) check that determines skip_meta.
        For LWT this is always False (result_metadata=None).
        """
        lwt = _make_lwt_insert_prepared(5)
        plain = _make_plain_insert_prepared(5)

        # Verify the difference
        assert bool(lwt.result_metadata) is False
        assert bool(plain.result_metadata) is True

        def check_skip_meta():
            return (
                bool(lwt.result_metadata),
                bool(plain.result_metadata),
            )

        benchmark(check_skip_meta)


# ============================================================================
# 7. MIXED TYPE BENCHMARKS (realistic LWT schemas)
# ============================================================================


def _make_lwt_mixed_type_prepared(protocol_version=4):
    """
    LWT INSERT with a realistic mixed-type schema:
    pk (UUID), name (text), age (int), email (text), active (boolean)
    """
    column_metadata = [
        ColumnMetadata("ks", "users", "pk", cqltypes.UUIDType),
        ColumnMetadata("ks", "users", "name", cqltypes.UTF8Type),
        ColumnMetadata("ks", "users", "age", cqltypes.Int32Type),
        ColumnMetadata("ks", "users", "email", cqltypes.UTF8Type),
        ColumnMetadata("ks", "users", "active", cqltypes.BooleanType),
    ]
    return PreparedStatement(
        column_metadata=column_metadata,
        query_id=b"lwt-mixed-id",
        routing_key_indexes=[0],
        query="INSERT INTO ks.users (pk, name, age, email, active) VALUES (?, ?, ?, ?, ?) IF NOT EXISTS",
        keyspace="ks",
        protocol_version=protocol_version,
        result_metadata=None,
        result_metadata_id=b"meta-mixed",
        is_lwt=True,
    )


class TestBindMixedTypes:
    """Benchmark bind() with realistic mixed-type LWT schemas."""

    @pytest.mark.benchmark(group="bind-mixed")
    def test_bind_lwt_mixed_types(self, benchmark):
        import uuid

        prepared = _make_lwt_mixed_type_prepared()
        values = (uuid.uuid4(), "John Doe", 30, "john@example.com", True)

        def do_bind():
            bs = BoundStatement(prepared)
            bs.bind(values)
            return bs

        result = benchmark(do_bind)
        assert len(result.values) == 5

    @pytest.mark.benchmark(group="bind-mixed")
    def test_bind_lwt_int32_only(self, benchmark):
        """Pure int32 for comparison with mixed types."""
        prepared = _make_lwt_insert_prepared(5)
        values = _int32_values(5)

        def do_bind():
            bs = BoundStatement(prepared)
            bs.bind(values)
            return bs

        result = benchmark(do_bind)
        assert len(result.values) == 5


class TestDecodeMixedTypes:
    """Benchmark decoding with realistic mixed-type LWT results."""

    @pytest.mark.benchmark(group="decode-mixed")
    def test_decode_lwt_not_applied_mixed(self, benchmark):
        """Not-applied result with mixed types."""
        import uuid

        test_uuid = uuid.uuid4()
        raw_rows = [
            [
                b"\x00",  # applied=False
                test_uuid.bytes,  # UUID
                b"John Doe",  # UTF8
                struct.pack(">i", 30),  # Int32
                b"john@example.com",  # UTF8
                b"\x01",  # Boolean
            ]
        ]
        coltypes = [
            cqltypes.BooleanType,
            cqltypes.UUIDType,
            cqltypes.UTF8Type,
            cqltypes.Int32Type,
            cqltypes.UTF8Type,
            cqltypes.BooleanType,
        ]

        result = benchmark(_decode_rows_python, raw_rows, coltypes)
        assert result[0][0] is False
        assert len(result[0]) == 6

    @pytest.mark.benchmark(group="decode-mixed")
    def test_decode_lwt_applied_mixed(self, benchmark):
        """Applied result (always just 1 boolean column regardless of schema)."""
        raw_rows = [[b"\x01"]]
        coltypes = [cqltypes.BooleanType]

        result = benchmark(_decode_rows_python, raw_rows, coltypes)
        assert result[0][0] is True
