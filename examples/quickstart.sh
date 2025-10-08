#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="${ROOT_DIR}/data/BL1_XFG_baseline.csv"
OUT_DIR="${ROOT_DIR}/examples/out"
BASELINE="${OUT_DIR}/baseline.jsonl"
ANALYZE_JSON="${OUT_DIR}/analyze.json"
SUGGEST_JSON="${OUT_DIR}/suggest.json"
REPORT_JSON="${OUT_DIR}/report.json"
PLOT_TXT="${OUT_DIR}/sense_index_plot.txt"

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Sample dataset not found: ${DATA_FILE}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

python -m tnfr_lfs.cli.tnfr_lfs_cli baseline "${BASELINE}" --simulate "${DATA_FILE}" --force
python -m tnfr_lfs.cli.tnfr_lfs_cli analyze "${BASELINE}" --export json > "${ANALYZE_JSON}"
python -m tnfr_lfs.cli.tnfr_lfs_cli suggest "${BASELINE}" --car-model XFG --track BL1 --export json > "${SUGGEST_JSON}"
python -m tnfr_lfs.cli.tnfr_lfs_cli report "${BASELINE}" --export json > "${REPORT_JSON}"
python -m tnfr_lfs.cli.tnfr_lfs_cli write-set "${BASELINE}" --car-model XFG --export markdown > "${OUT_DIR}/setup_plan.md"

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
