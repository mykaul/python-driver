#!/usr/bin/env bash
# setup_bench.sh -- One-time setup for the vector ingestion benchmark.
#
# What it does:
#   1. Starts a ScyllaDB container (podman) with vector-search support
#   2. Creates three Python virtual environments (using uv):
#        venv-baseline  -- stock scylla-driver from PyPI (Variant A)
#        venv-enhanced  -- scylla-driver from perf/all-merged branch (Variants B, C)
#        venv-rs-driver -- python-rs-driver (Variant D)
#   3. Downloads the Cohere 768-dim 1M dataset (once, skipped if already present)
#
# Usage:
#   bash bench/setup_bench.sh [--skip-container] [--skip-venvs] [--skip-dataset]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$SCRIPT_DIR"

CONTAINER_NAME="scylla-bench"
CONTAINER_IMAGE="scylladb/scylla:latest"
CONTAINER_SMP=4
CONTAINER_MEMORY="8G"
CONTAINER_CPUSET="0-3"       # Pin ScyllaDB to cores 0-3
CLIENT_CPUSET="4-5"          # Pin client benchmark to cores 4-5
CQL_PORT=9042

DATASET_DIR="$HOME/vector_bench/dataset/cohere/cohere_medium_1m"

SKIP_CONTAINER=false
SKIP_VENVS=false
SKIP_DATASET=false

for arg in "$@"; do
    case "$arg" in
        --skip-container) SKIP_CONTAINER=true ;;
        --skip-venvs)     SKIP_VENVS=true ;;
        --skip-dataset)   SKIP_DATASET=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Verify uv is available
# --------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' is not installed. Install it first: https://docs.astral.sh/uv/"
    exit 1
fi
echo "Using uv $(uv --version)"

# --------------------------------------------------------------------------
# 1. ScyllaDB container
# --------------------------------------------------------------------------
setup_container() {
    echo "=== Setting up ScyllaDB container ==="

    # Check for podman or docker
    if command -v podman &>/dev/null; then
        RUNTIME=podman
    elif command -v docker &>/dev/null; then
        RUNTIME=docker
        echo "WARNING: podman not found, falling back to docker"
    else
        echo "ERROR: Neither podman nor docker found. Install podman first."
        exit 1
    fi

    # Stop existing container if running
    if $RUNTIME ps -a --format '{{.Names}}' 2>/dev/null | grep -qw "$CONTAINER_NAME"; then
        echo "Stopping existing container '$CONTAINER_NAME'..."
        $RUNTIME rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi

    echo "Starting ScyllaDB container (pinned to CPUs $CONTAINER_CPUSET, production mode)..."
    $RUNTIME run -d \
        --name "$CONTAINER_NAME" \
        --cpuset-cpus "$CONTAINER_CPUSET" \
        -p "${CQL_PORT}:${CQL_PORT}" \
        -p "19042:19042" \
        "$CONTAINER_IMAGE" \
        --smp "$CONTAINER_SMP" \
        --memory "$CONTAINER_MEMORY" \
        --unsafe-bypass-fsync 1 \
        --developer-mode 0

    echo "Waiting for CQL readiness (this may take 30-60 seconds)..."
    local retries=0
    local max_retries=90
    while ! $RUNTIME exec "$CONTAINER_NAME" cqlsh -e "SELECT now() FROM system.local" &>/dev/null; do
        retries=$((retries + 1))
        if [ "$retries" -ge "$max_retries" ]; then
            echo "ERROR: ScyllaDB did not become ready within ${max_retries} seconds"
            $RUNTIME logs "$CONTAINER_NAME" 2>&1 | tail -20
            exit 1
        fi
        sleep 1
    done
    echo "ScyllaDB is ready on port $CQL_PORT (took ~${retries}s)"
}

# --------------------------------------------------------------------------
# 2. Python virtual environments
# --------------------------------------------------------------------------
setup_venvs() {
    echo ""
    echo "=== Setting up Python virtual environments ==="

    local common_deps=(numpy pyarrow s3fs polars tqdm)

    # --- venv-baseline: stock scylla-driver from PyPI ---
    if [ ! -d "$BENCH_DIR/venv-baseline" ]; then
        echo "Creating venv-baseline (stock scylla-driver from PyPI)..."
        uv venv "$BENCH_DIR/venv-baseline"
        uv pip install --python "$BENCH_DIR/venv-baseline/bin/python3" \
            "${common_deps[@]}" scylla-driver
        echo "venv-baseline ready."
    else
        echo "venv-baseline already exists, skipping."
    fi

    # --- venv-enhanced: scylla-driver from local perf/all-merged branch ---
    if [ ! -d "$BENCH_DIR/venv-enhanced" ]; then
        echo "Creating venv-enhanced (local perf/all-merged branch with Cython)..."
        uv venv "$BENCH_DIR/venv-enhanced"
        uv pip install --python "$BENCH_DIR/venv-enhanced/bin/python3" \
            "${common_deps[@]}" cython setuptools

        # Build Cython extensions explicitly, then install in editable mode
        echo "Building Cython extensions and installing enhanced driver..."
        (
            cd "$REPO_ROOT"
            "$BENCH_DIR/venv-enhanced/bin/python3" setup.py build_ext --inplace 2>&1 | tail -5
            uv pip install --python "$BENCH_DIR/venv-enhanced/bin/python3" -e .
        )
        # Verify Cython extensions were built
        local so_count
        so_count=$(find "$REPO_ROOT/cassandra" -name '*.so' | wc -l)
        echo "  Cython .so files built: $so_count"
        echo "venv-enhanced ready."
    else
        echo "venv-enhanced already exists, skipping."
    fi

    # --- venv-rs-driver: python-rs-driver (Rust-backed) ---
    if [ ! -d "$BENCH_DIR/venv-rs-driver" ]; then
        echo "Creating venv-rs-driver (python-rs-driver from GitHub)..."
        uv venv "$BENCH_DIR/venv-rs-driver"
        uv pip install --python "$BENCH_DIR/venv-rs-driver/bin/python3" \
            "${common_deps[@]}"
        echo "Installing python-rs-driver (this compiles Rust code, may take a few minutes)..."
        uv pip install --python "$BENCH_DIR/venv-rs-driver/bin/python3" \
            "scylla @ git+https://github.com/scylladb-zpp-2025-python-rs-driver/python-rs-driver.git"
        echo "venv-rs-driver ready."
    else
        echo "venv-rs-driver already exists, skipping."
    fi
}

# --------------------------------------------------------------------------
# 3. Dataset download
# --------------------------------------------------------------------------
download_dataset() {
    echo ""
    echo "=== Downloading Cohere 768-dim 1M dataset ==="

    local train_file="$DATASET_DIR/shuffle_train.parquet"
    local url="https://assets.zilliz.com/benchmark/cohere_medium_1m/shuffle_train.parquet"

    if [ -f "$train_file" ]; then
        # Quick integrity check: parquet files end with "PAR1" magic bytes
        if tail -c 4 "$train_file" | grep -q "PAR1"; then
            echo "Dataset already exists and looks valid, skipping download."
            ls -lh "$train_file"
            return
        else
            echo "Existing file is corrupt, re-downloading..."
            rm -f "$train_file"
        fi
    fi

    mkdir -p "$DATASET_DIR"

    echo "Downloading from $url (~2.9 GB, may take a few minutes)..."
    if command -v wget &>/dev/null; then
        wget -O "$train_file" "$url"
    elif command -v curl &>/dev/null; then
        curl -L -o "$train_file" "$url"
    else
        echo "ERROR: Neither wget nor curl found."
        exit 1
    fi

    # Verify
    if ! tail -c 4 "$train_file" | grep -q "PAR1"; then
        echo "ERROR: Downloaded file does not appear to be a valid parquet file."
        exit 1
    fi

    echo "Download complete:"
    ls -lh "$train_file"
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
echo "============================================="
echo " Vector Ingestion Benchmark -- Setup"
echo "============================================="
echo "Repo root:     $REPO_ROOT"
echo "Bench dir:     $BENCH_DIR"
echo "Dataset dir:   $DATASET_DIR"
echo "ScyllaDB CPUs: $CONTAINER_CPUSET (smp=$CONTAINER_SMP, no overprovisioned, unsafe-bypass-fsync)"
echo "Client CPUs:   $CLIENT_CPUSET"
echo ""

if ! $SKIP_DATASET; then
    download_dataset
else
    echo "=== Skipping dataset download (--skip-dataset) ==="
fi

if ! $SKIP_VENVS; then
    setup_venvs
else
    echo "=== Skipping venv setup (--skip-venvs) ==="
fi

if ! $SKIP_CONTAINER; then
    setup_container
else
    echo "=== Skipping container setup (--skip-container) ==="
fi

echo ""
echo "============================================="
echo " Setup complete!"
echo "============================================="
echo ""
echo "Next step: bash bench/run_bench.sh"
