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
Benchmarks for apply_parameters() with and without caching.

apply_parameters() creates parameterized CQL type classes (e.g.
ListType(UTF8Type), MapType(UTF8Type, Int32Type)) using type(), which
is expensive. Caching makes repeated calls return the same class object
(stable singletons), eliminating the type() overhead.

Run with:
    pytest benchmarks/test_apply_parameters_benchmark.py -v
"""

import pytest

from cassandra import cqltypes


# ---------------------------------------------------------------------------
# Reference: original uncached implementation (copied from master)
# ---------------------------------------------------------------------------


def apply_parameters_uncached(cls, subtypes, names=None):
    """Original apply_parameters without caching (baseline)."""
    if cls.num_subtypes != "UNKNOWN" and len(subtypes) != cls.num_subtypes:
        raise ValueError(
            "%s types require %d subtypes (%d given)"
            % (cls.typename, cls.num_subtypes, len(subtypes))
        )
    newname = cls.cass_parameterized_type_with(subtypes)
    return type(
        newname,
        (cls,),
        {"subtypes": subtypes, "cassname": cls.cassname, "fieldnames": names},
    )


# ---------------------------------------------------------------------------
# Test type combinations (representative of real workloads)
# ---------------------------------------------------------------------------

SIMPLE_TYPES = [
    cqltypes.UTF8Type,
    cqltypes.Int32Type,
    cqltypes.LongType,
    cqltypes.FloatType,
    cqltypes.DoubleType,
    cqltypes.BooleanType,
    cqltypes.TimestampType,
    cqltypes.UUIDType,
    cqltypes.BytesType,
    cqltypes.DecimalType,
]

# Parameterized type specs: (base_class, subtypes_tuple)
PARAM_TYPE_SPECS = [
    (cqltypes.ListType, (cqltypes.UTF8Type,)),
    (cqltypes.ListType, (cqltypes.Int32Type,)),
    (cqltypes.SetType, (cqltypes.UTF8Type,)),
    (cqltypes.SetType, (cqltypes.UUIDType,)),
    (cqltypes.MapType, (cqltypes.UTF8Type, cqltypes.Int32Type)),
    (cqltypes.MapType, (cqltypes.UTF8Type, cqltypes.UTF8Type)),
    (cqltypes.FrozenType, (cqltypes.ListType.apply_parameters((cqltypes.UTF8Type,)),)),
    (cqltypes.ReversedType, (cqltypes.TimestampType,)),
]


# ---------------------------------------------------------------------------
# Benchmark: Single apply_parameters call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_cls,subtypes",
    [
        (cqltypes.ListType, (cqltypes.UTF8Type,)),
        (cqltypes.MapType, (cqltypes.UTF8Type, cqltypes.Int32Type)),
        (cqltypes.SetType, (cqltypes.UUIDType,)),
        (cqltypes.FrozenType, (cqltypes.TimestampType,)),
    ],
    ids=["List<UTF8>", "Map<UTF8,Int32>", "Set<UUID>", "Frozen<Timestamp>"],
)
def test_apply_parameters_uncached(benchmark, base_cls, subtypes):
    """Baseline: call type() every time (original code path)."""
    benchmark(apply_parameters_uncached, base_cls, subtypes)


@pytest.mark.parametrize(
    "base_cls,subtypes",
    [
        (cqltypes.ListType, (cqltypes.UTF8Type,)),
        (cqltypes.MapType, (cqltypes.UTF8Type, cqltypes.Int32Type)),
        (cqltypes.SetType, (cqltypes.UUIDType,)),
        (cqltypes.FrozenType, (cqltypes.TimestampType,)),
    ],
    ids=["List<UTF8>", "Map<UTF8,Int32>", "Set<UUID>", "Frozen<Timestamp>"],
)
def test_apply_parameters_cached(benchmark, base_cls, subtypes):
    """Cached: dict lookup on hit (new code path)."""
    # Warm the cache
    base_cls.apply_parameters(subtypes)

    benchmark(base_cls.apply_parameters, subtypes)


# ---------------------------------------------------------------------------
# Benchmark: Batch of apply_parameters (simulating read_type for a result set)
# ---------------------------------------------------------------------------


def _batch_uncached(specs):
    """Apply parameters for a batch of type specs without caching."""
    return [apply_parameters_uncached(cls, st) for cls, st in specs]


def _batch_cached(specs):
    """Apply parameters for a batch of type specs with caching."""
    return [cls.apply_parameters(st) for cls, st in specs]


@pytest.mark.parametrize("n_specs", [4, 8])
def test_batch_apply_uncached(benchmark, n_specs):
    """Batch: build N parameterized types without caching."""
    specs = PARAM_TYPE_SPECS[:n_specs]
    benchmark(_batch_uncached, specs)


@pytest.mark.parametrize("n_specs", [4, 8])
def test_batch_apply_cached(benchmark, n_specs):
    """Batch: build N parameterized types with caching."""
    specs = PARAM_TYPE_SPECS[:n_specs]
    # Warm
    _batch_cached(specs)
    benchmark(_batch_cached, specs)


# ---------------------------------------------------------------------------
# Benchmark: Simulated read_type column parsing for a typical result set
# with mixed simple and parameterized types
# ---------------------------------------------------------------------------


def _simulate_metadata_uncached():
    """Simulate parsing column metadata for a 10-column result set."""
    types = []
    # 5 simple types (no apply_parameters needed)
    types.extend(SIMPLE_TYPES[:5])
    # 5 parameterized types
    types.append(apply_parameters_uncached(cqltypes.ListType, (cqltypes.UTF8Type,)))
    types.append(
        apply_parameters_uncached(
            cqltypes.MapType, (cqltypes.UTF8Type, cqltypes.Int32Type)
        )
    )
    types.append(apply_parameters_uncached(cqltypes.SetType, (cqltypes.UUIDType,)))
    types.append(
        apply_parameters_uncached(cqltypes.FrozenType, (cqltypes.TimestampType,))
    )
    types.append(apply_parameters_uncached(cqltypes.ReversedType, (cqltypes.LongType,)))
    return types


def _simulate_metadata_cached():
    """Simulate parsing column metadata for a 10-column result set (cached)."""
    types = []
    types.extend(SIMPLE_TYPES[:5])
    types.append(cqltypes.ListType.apply_parameters((cqltypes.UTF8Type,)))
    types.append(
        cqltypes.MapType.apply_parameters((cqltypes.UTF8Type, cqltypes.Int32Type))
    )
    types.append(cqltypes.SetType.apply_parameters((cqltypes.UUIDType,)))
    types.append(cqltypes.FrozenType.apply_parameters((cqltypes.TimestampType,)))
    types.append(cqltypes.ReversedType.apply_parameters((cqltypes.LongType,)))
    return types


def test_simulate_metadata_uncached(benchmark):
    """Simulate metadata parsing for 10-col result set — uncached."""
    benchmark(_simulate_metadata_uncached)


def test_simulate_metadata_cached(benchmark):
    """Simulate metadata parsing for 10-col result set — cached."""
    # Warm cache
    _simulate_metadata_cached()
    benchmark(_simulate_metadata_cached)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_cached_returns_same_object():
    """Cached apply_parameters returns the exact same type object."""
    a = cqltypes.ListType.apply_parameters((cqltypes.UTF8Type,))
    b = cqltypes.ListType.apply_parameters((cqltypes.UTF8Type,))
    assert a is b


def test_different_params_different_types():
    """Different subtypes produce different cached types."""
    a = cqltypes.ListType.apply_parameters((cqltypes.UTF8Type,))
    b = cqltypes.ListType.apply_parameters((cqltypes.Int32Type,))
    assert a is not b
    assert a.subtypes == (cqltypes.UTF8Type,)
    assert b.subtypes == (cqltypes.Int32Type,)


def test_cached_type_attributes():
    """Cached type has correct attributes."""
    t = cqltypes.MapType.apply_parameters((cqltypes.UTF8Type, cqltypes.LongType))
    assert t.subtypes == (cqltypes.UTF8Type, cqltypes.LongType)
    assert issubclass(t, cqltypes.MapType)
    assert t.cassname == cqltypes.MapType.cassname


def test_cached_matches_uncached():
    """Cached version produces equivalent types to uncached."""
    for cls, subtypes in PARAM_TYPE_SPECS:
        cached = cls.apply_parameters(subtypes)
        uncached = apply_parameters_uncached(cls, subtypes)

        assert cached.subtypes == uncached.subtypes
        assert cached.cassname == uncached.cassname
        assert issubclass(cached, cls)
        assert issubclass(uncached, cls)


def test_nested_parameterized_types():
    """Nested parameterized types (e.g. List<Map<text,int>>) are cached."""
    inner = cqltypes.MapType.apply_parameters((cqltypes.UTF8Type, cqltypes.Int32Type))
    outer1 = cqltypes.ListType.apply_parameters((inner,))
    outer2 = cqltypes.ListType.apply_parameters((inner,))
    assert outer1 is outer2
    assert outer1.subtypes == (inner,)
