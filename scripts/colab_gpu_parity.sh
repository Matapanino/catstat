#!/usr/bin/env bash
# Run catstat's CPU/GPU parity + GPU benchmark on a Google Colab GPU VM, and pull back the
# results. There is no local GPU, so this is the ONLY way to exercise backend="gpu".
#
# Mirrors the repleafgbm Colab dev loop. Requires the Colab CLI:
#   uv tool install google-colab-cli   # or: pip install google-colab-cli
#
# Usage:
#   bash scripts/colab_gpu_parity.sh [--gpu T4|L4|A100] [--session NAME] [--keep]
#                                    [--include-untracked]
#
# By default the upload tarball is `git archive HEAD` (committed code only) and a
# dirty tracked tree is refused. Pass --include-untracked to pack the live working
# tree (in-progress work), minus a strict exclude list and with secrets refused.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="catstat-gpu"
KEEP=0
INCLUDE_UNTRACKED=0
# Wall-clock cap on the remote exec; `colab exec` can hang for ages if the kernel websocket
# drops, so an external watchdog SIGKILLs it. The parity job is a few minutes; this is headroom.
EXEC_TIMEOUT="${EXEC_TIMEOUT:-1200}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --include-untracked) INCLUDE_UNTRACKED=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if ! command -v colab >/dev/null 2>&1; then
    echo "error: the 'colab' CLI is not installed." >&2
    echo "  uv tool install google-colab-cli   # or: pip install google-colab-cli" >&2
    exit 1
fi

DATE="$(date +%F)"
REPORT_OUT="docs/verdicts/${DATE}-gpu-parity-report.md"
JSONL_OUT="benchmarks/results/${DATE}-${GPU}-gpu-parity.jsonl"
# --- release-safe upload tarball ---------------------------------------------
# Default: archive the COMMITTED tree (git archive HEAD); untracked files never
# ride along and uncommitted tracked changes are refused, so secrets / caches /
# build artifacts / private notes cannot leak into the upload. --include-untracked
# packs the live working tree instead (for in-progress work), minus the excludes
# below and with a hard refusal on secret-like files.
TAR_EXCLUDES=(
    --exclude='./.git' --exclude='**/.git'
    --exclude='**/.claude'
    --exclude='**/.env' --exclude='**/.env.*'
    --exclude='**/*.pem' --exclude='**/*.key' --exclude='**/id_rsa*'
    --exclude='**/.pypirc' --exclude='**/.npmrc'
    --exclude='**/__pycache__' --exclude='*.egg-info'
    --exclude='**/.pytest_cache' --exclude='**/.mypy_cache' --exclude='**/.ruff_cache'
    --exclude='**/.coverage'
    --exclude='./artifacts' --exclude='**/catboost_info' --exclude='./site'
    --exclude='./dist' --exclude='./build' --exclude='./target' --exclude='**/target'
    --exclude='./docs/paper' --exclude='./docs/gpu-research'
    --exclude='**/*.jsonl'                  # benchmark result ledgers
    --exclude='**/next-session-prompt.md'   # private runbook (assistant prompt)
)

guard_no_secrets() {
    local hits
    hits="$(find . -path ./.git -prune -o \( -name '.env' -o -name '.env.*' \
        -o -name '*.pem' -o -name '*.key' -o -name 'id_rsa*' \
        -o -name '.pypirc' -o -name '.npmrc' \) -print 2>/dev/null)"
    if [[ -n "$hits" ]]; then
        echo "error: refusing to pack -- secret-like files are present:" >&2
        echo "$hits" >&2
        exit 1
    fi
}

make_tarball() {  # $1 = output path
    if [[ "$INCLUDE_UNTRACKED" -eq 1 ]]; then
        echo ">> packing live working tree (--include-untracked) -> $1"
        guard_no_secrets
        tar "${TAR_EXCLUDES[@]}" -czf "$1" .
        return
    fi
    if ! git diff --quiet HEAD; then
        echo "error: tracked files differ from HEAD -- 'git archive HEAD' would omit your changes." >&2
        echo "  commit/stash them, or re-run with --include-untracked to pack the live tree." >&2
        git status --short >&2
        exit 1
    fi
    if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        echo ">> note: untracked files are NOT included (use --include-untracked to add them):"
        git ls-files --others --exclude-standard
    fi
    echo ">> archiving committed tree (git archive HEAD) -> $1"
    git archive --format=tar.gz -o "$1" HEAD
}

TARBALL="$(mktemp -t catstat-XXXXXX).tar.gz"
cleanup_local() { rm -f "$TARBALL"; }
trap cleanup_local EXIT

make_tarball "$TARBALL"

echo ">> provisioning $GPU VM (session: $SESSION)"
colab new -s "$SESSION" --gpu "$GPU"
stop_vm() { [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap 'cleanup_local; stop_vm' EXIT

echo ">> uploading working tree"
colab upload -s "$SESSION" "$TARBALL" /content/catstat.tar.gz

echo ">> running CPU/GPU parity on the GPU (watchdog: ${EXEC_TIMEOUT}s)"
colab exec -s "$SESSION" --timeout 1100 -f scripts/colab_gpu_parity.py &
exec_pid=$!
( sleep "$EXEC_TIMEOUT"; kill -KILL "$exec_pid" 2>/dev/null ) &
wd_pid=$!
exec_rc=0
wait "$exec_pid" || exec_rc=$?
kill "$wd_pid" 2>/dev/null || true
wait "$wd_pid" 2>/dev/null || true

if [[ "$exec_rc" -ne 0 ]]; then
    echo ">> colab exec failed or timed out (rc=$exec_rc); skipping downloads." >&2
    exit "$exec_rc"   # EXIT trap still stops the VM
fi

echo ">> downloading parity report -> $REPORT_OUT"
mkdir -p docs/verdicts benchmarks/results
colab download -s "$SESSION" /content/parity_report.md "$REPORT_OUT"
echo ">> downloading parity JSONL -> $JSONL_OUT"
colab download -s "$SESSION" /content/parity.jsonl "$JSONL_OUT" || echo "   (no JSONL produced)"

echo ">> done. report at $REPORT_OUT"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
