#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RELCO_REPO_DIR="${ROOT_DIR}"
export RELCO_OUT_DIR="${RELCO_OUT_DIR:-${ROOT_DIR}/out}"
export RELCO_CHECKPOINT_DIR="${RELCO_CHECKPOINT_DIR:-${RELCO_OUT_DIR}/checkpoints}"
export RELCO_DATA_DIR="${RELCO_DATA_DIR:-${ROOT_DIR}/data/datasets}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/tests${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RELCO_OUT_DIR}" "${RELCO_CHECKPOINT_DIR}" "${RELCO_DATA_DIR}"

echo "Exported RELCO_DATA_DIR=${RELCO_DATA_DIR}"
echo "Exported RELCO_OUT_DIR=${RELCO_OUT_DIR}"
echo "Exported RELCO_CHECKPOINT_DIR=${RELCO_CHECKPOINT_DIR}"
echo "Exported PYTHONPATH=${PYTHONPATH}"
