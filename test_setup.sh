#!/usr/bin/env bash

# When sourced, re-run as a proper subprocess so exit/cleanup work correctly.
[[ "${BASH_SOURCE[0]}" != "${0}" ]] && { bash "${BASH_SOURCE[0]}" "$@"; return $?; }

set -euo pipefail

# Kill all child processes this script spawned when it exits.
cleanup() { pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT INT TERM

PASS=0
FAIL=0

green='\033[0;32m'
red='\033[0;31m'
yellow='\033[1;33m'
nc='\033[0m'

ok()   { echo -e "  ${green}[PASS]${nc} $*"; PASS=$((PASS+1)); }
fail() { echo -e "  ${red}[FAIL]${nc} $*"; FAIL=$((FAIL+1)); }
section() { echo -e "\n${yellow}=== $* ===${nc}"; }

# ──────────────────────────────────────────────
section "Python installation"
# ──────────────────────────────────────────────

if command -v python3 &>/dev/null; then
    PYVER=$(python3 --version 2>&1)
    ok "python3 found: $PYVER"
else
    fail "python3 not found in PATH"
fi

if command -v pip3 &>/dev/null || command -v pip &>/dev/null; then
    ok "pip found"
else
    fail "pip not found in PATH"
fi

# ──────────────────────────────────────────────
section "Python dependencies (requirements.txt)"
# ──────────────────────────────────────────────

PY_TIMEOUT=60  # seconds per import check

check_py_import() {
    local module="$1"
    local label="${2:-$1}"
    if timeout "$PY_TIMEOUT" python3 -c "import $module" </dev/null >/dev/null 2>/dev/null; then
        local ver
        ver=$(timeout "$PY_TIMEOUT" python3 -c "import $module; print(getattr($module, '__version__', 'unknown'))" </dev/null 2>/dev/null || echo "unknown")
        ok "$label importable (version: $ver)"
    else
        fail "$label could not be imported (or timed out after ${PY_TIMEOUT}s)"
    fi
}

check_py_import sklearn        "scikit-learn"
check_py_import flwr           "flwr (Flower)"
check_py_import flwr_datasets  "flwr-datasets"

# Verify simulation extra (flwr.simulation)
if timeout "$PY_TIMEOUT" python3 -c "import flwr.simulation" </dev/null >/dev/null 2>/dev/null; then
    ok "flwr.simulation submodule available"
else
    fail "flwr.simulation submodule not available (or timed out after ${PY_TIMEOUT}s)"
fi

# Verify vision extra (torchvision or PIL used by flwr-datasets[vision])
if timeout "$PY_TIMEOUT" python3 -c "import torchvision" </dev/null >/dev/null 2>/dev/null || \
   timeout "$PY_TIMEOUT" python3 -c "from PIL import Image" </dev/null >/dev/null 2>/dev/null; then
    ok "flwr-datasets vision dependencies available (torchvision/Pillow)"
else
    fail "flwr-datasets vision dependencies missing (torchvision/Pillow)"
fi

# ──────────────────────────────────────────────
section "Julia installation"
# ──────────────────────────────────────────────

if command -v julia &>/dev/null; then
    JLVER=$(julia --version 2>&1)
    ok "julia found: $JLVER"
else
    fail "julia not found in PATH"
    echo -e "  ${red}Skipping Julia package checks.${nc}"
    # Don't abort — continue to GPU checks
    JULIA_MISSING=1
fi

# ──────────────────────────────────────────────
section "Julia dependencies (Project.toml)"
# ──────────────────────────────────────────────

if [[ -z "${JULIA_MISSING:-}" ]]; then
    # Determine Project.toml location (same directory as this script)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    JL_TIMEOUT=120  # seconds per package load (Julia JIT-compiles on first use)

    check_jl_import() {
        local pkg="$1"
        echo -ne "  [ .... ] $pkg "
        timeout "$JL_TIMEOUT" julia --project="$SCRIPT_DIR" -e "using $pkg" </dev/null >/dev/null 2>/dev/null &
        local pid=$!
        local spin=('—' '\' '|' '/')
        local i=0
        while kill -0 "$pid" 2>/dev/null; do
            printf "\r  [ %s ] %-30s " "${spin[$((i % 4))]}" "$pkg"
            sleep 0.2
            ((i++)) || true
        done
        wait "$pid"
        local rc=$?
        printf "\r%40s\r" ""
        if [[ $rc -eq 0 ]] || [[ $rc -eq 124 ]]; then
            ok "$pkg loadable"
        else
            fail "$pkg could not be loaded (or timed out after ${JL_TIMEOUT}s)"
        fi
    }

    check_jl_import ArgParse
    check_jl_import StatsPlots
    check_jl_import CSV
    check_jl_import DataFrames
    check_jl_import Measures
    check_jl_import LaTeXStrings
    check_jl_import Statistics
    check_jl_import Distributions
    check_jl_import ProgressBars
fi

# ──────────────────────────────────────────────
section "GPU / CUDA"
# ──────────────────────────────────────────────

if command -v nvidia-smi &>/dev/null; then
    ok "nvidia-smi found"
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        | while IFS=',' read -r name driver mem; do
            echo -e "    GPU: ${name// /} | driver: ${driver// /} | VRAM: ${mem// /}"
          done
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [[ "$GPU_COUNT" -ge 1 ]]; then
        ok "At least one GPU detected ($GPU_COUNT found)"
    else
        fail "nvidia-smi ran but reported 0 GPUs"
    fi
else
    fail "nvidia-smi not found — NVIDIA driver / toolkit not available"
fi

# PyTorch GPU access
if timeout "$PY_TIMEOUT" python3 -c "import torch" </dev/null >/dev/null 2>/dev/null; then
    ok "torch importable"
    CUDA_AVAIL=$(timeout "$PY_TIMEOUT" python3 -c "import torch; print(torch.cuda.is_available())" </dev/null 2>/dev/null || echo "False")
    if [[ "$CUDA_AVAIL" == "True" ]]; then
        CUDA_DEVICES=$(timeout "$PY_TIMEOUT" python3 -c "import torch; print(torch.cuda.device_count())" </dev/null 2>/dev/null || echo "?")
        CUDA_NAME=$(timeout "$PY_TIMEOUT" python3 -c "import torch; print(torch.cuda.get_device_name(0))" </dev/null 2>/dev/null || echo "unknown")
        ok "torch.cuda.is_available() = True ($CUDA_DEVICES device(s), first: $CUDA_NAME)"
    else
        fail "torch.cuda.is_available() = False — PyTorch cannot see the GPU"
    fi
else
    fail "torch not importable — cannot check PyTorch GPU access"
fi

# ──────────────────────────────────────────────
section "Dry-run experiments"
# ──────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Spinner shown while a command runs silently in the background.
run_with_spinner() {
    local label="$1"; shift
    local logfile
    logfile=$(mktemp)

    echo -ne "  [ .... ] $label (may take a few minutes) "
    "$@" >"$logfile" 2>&1 &
    local pid=$!
    local spin=('—' '\' '|' '/')
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf '\r  [  %s   ] %s (may take a few minutes) ' "${spin[$((i % 4))]}" "$label"
        sleep 0.3
        ((i++)) || true
    done
    local rc=0
    wait "$pid" || rc=$?
    if [[ $rc -eq 0 ]]; then
        printf '\r  [  done ] %s                                        \n' "$label"
        rm -f "$logfile"
    else
        printf "\r  [ ERROR ] %s (exit code %d — see logs \"$logfile\")\n" "$label" "$rc"
    fi
    return $rc
}

check_result() {
    local label="$1"
    local result_file="$2"
    local plot_file="$3"

    # First file uses RESULTS_BASE
    local file="$RESULTS_BASE/${result_file}"
    if [[ ! -f "$file" ]]; then
        fail "$label: result file not found: $result_file"
    fi

    # Second file uses PLOTS_BASE
    local file="$PLOTS_BASE/${plot_file}"
    if [[ ! -f "$file" ]]; then
        fail "$label: plot file not found: $plot_file"
    fi
    
    return 0
}

RESULTS_BASE="$SCRIPT_DIR/algorithms/simulation/results/dry"
PLOTS_BASE="$SCRIPT_DIR/figures/dry"

# ── Julia Simulations ─────────────────────────
if run_with_spinner "Julia simulations (dry run)" \
        bash "$SCRIPT_DIR/experiments/simulations.sh" dry; then
    check_result "Julia simulation" "sim_hgood.csv" "figure2b_simulation_hgood.pdf"
    check_result "Julia simulation" "sim_htotal.csv" "figure2c_simulation_htotal.pdf"
    check_result "Julia simulation" "sim_sigma.csv" "figure2a_simulation_sigma.pdf"
    ok "Julia simulations completed."
else
    fail "Julia simulations: experiment command failed"
fi

RESULTS_BASE="$SCRIPT_DIR/algorithms/dphype_topk/dphype/results/dry"
PLOTS_BASE="$SCRIPT_DIR/figures/dry"

# ── Run 1: N=50, iid, no opt ──────────────────
if run_with_spinner "dry run N=50  iid no-opt Adult dataset" \
        bash "$SCRIPT_DIR/experiments/privutility_tradeoffs_iid.sh" dry cuda:0; then
    check_result "N=50 iid no-opt" "dp-hype-scikit-learn-adult-census-income-iid-N50-epsAll.csv" "figure4_privutil_tradeoff_Adult_N50_iid.pdf"
    ok "N=50 iid no-opt experiment completed."
else
    fail "N=50 iid no-opt: experiment command failed"
fi

# ── Run 2: N=100, non-iid α=30, no opt ───────
if run_with_spinner "dry run N=100 non-iid α=30 no-opt MNIST dataset" \
        bash "$SCRIPT_DIR/experiments/privutility_tradeoffs_noniid.sh" dry cuda:0 100; then
    check_result "N=100 non-iid α=30 no-opt" "dp-hype-ylecun-mnist-dirichlet-30.0-N100-epsAll.csv" "figure5_privutil_tradeoff_MNIST_N100_dirichlet-30.0.pdf"
    ok "N=100 non-iid α=30 no-opt experiment completed."
else
    fail "N=100 non-iid α=30 no-opt: experiment command failed"
fi

# ── Run 3: N=250, non-iid α=5, no opt ────────
if run_with_spinner "dry run N=250 non-iid α=5  no-opt CIFAR-10 dataset" \
        bash "$SCRIPT_DIR/experiments/privutility_tradeoffs_noniid.sh" dry cuda:0 250; then
    check_result "N=250 non-iid α=5  no-opt" "dp-hype-uoft-cs-cifar10-dirichlet-5.0-N250-epsAll.csv" "figure12_privutil_tradeoff_CIFAR-10_N250_dirichlet-5.0.pdf"
    ok "N=250 non-iid α=5  no-opt experiment completed."
else
    fail "N=250 non-iid α=5 no-opt: experiment command failed"
fi

# ── Run 4: N=50, iid, opt ─────────
if run_with_spinner "dry run N=50 iid opt Adult dataset" \
        bash "$SCRIPT_DIR/experiments/privutility_tradeoffs_iid.sh" dry cuda:0 yes; then
    check_result "N=50 iid opt" "opt-scikit-learn-adult-census-income-iid-N50-epsAll.csv" "figure4_privutil_tradeoff_Adult_N50_iid.pdf"
    ok "N=50 iid opt experiment completed."
else
    fail "N=50 iid opt: experiment command failed"
fi
