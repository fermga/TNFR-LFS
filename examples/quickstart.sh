#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
DATA_FILE="$(python - <<'PY'
from tnfr_lfs.examples.quickstart_dataset import dataset_path

print(dataset_path())
PY
)"
OUT_DIR="${ROOT_DIR}/examples/out"
BASELINE="${OUT_DIR}/baseline.jsonl"
ANALYZE_JSON="${OUT_DIR}/analyze.json"
SUGGEST_JSON="${OUT_DIR}/suggest.json"
REPORT_JSON="${OUT_DIR}/report.json"
PLOT_TXT="${OUT_DIR}/sense_index_plot.txt"

if [[ -n "${CLI_BIN:-}" ]]; then
    # shellcheck disable=SC2206
    CLI_CMD=(${CLI_BIN})
elif command -v tnfr_lfs >/dev/null 2>&1; then
    CLI_CMD=(tnfr_lfs)
else
    CLI_CMD=(python -m tnfr_lfs)
fi

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Sample dataset not found: ${DATA_FILE}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

"${CLI_CMD[@]}" baseline "${BASELINE}" --simulate "${DATA_FILE}" --force
"${CLI_CMD[@]}" analyze "${BASELINE}" --export json > "${ANALYZE_JSON}"
"${CLI_CMD[@]}" suggest "${BASELINE}" --car-model XFG --track BL1 --export json > "${SUGGEST_JSON}"
"${CLI_CMD[@]}" report "${BASELINE}" --export json > "${REPORT_JSON}"
"${CLI_CMD[@]}" write-set "${BASELINE}" --car-model XFG --export markdown > "${OUT_DIR}/setup_plan.md"

python <<'PY'
import json
from pathlib import Path

analysis_path = Path("${ANALYZE_JSON}")
output_path = Path("${PLOT_TXT}")
if analysis_path.exists():
    payload = json.loads(analysis_path.read_text(encoding="utf8"))
    series = payload.get("series", [])
    values = [item.get("sense_index", 0.0) for item in series if isinstance(item, dict)]
    if values:
        max_value = max(values) or 1.0
        lines = [
            f"t={idx:02d} |" + "█" * int((value / max_value) * 40)
            for idx, value in enumerate(values)
        ]
    else:
        lines = ["No Sense Index samples available"]
else:
    lines = ["Analysis file not found"]
output_path.write_text("\n".join(lines), encoding="utf8")
PY

echo "Artefacts generated in ${OUT_DIR}"
