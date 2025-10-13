SHELL := /bin/bash

.PHONY: quickstart install dev-install benchmark-delta-cache report-similar-tests report-similar-tests-sweep verify-tnfr-parity
quickstart:
	@if [[ ! -f examples/quickstart.sh ]]; then \
		echo "Error: quickstart script examples/quickstart.sh not found. Please pull the examples directory or download the script before rerunning quickstart."; \
		exit 1; \
	fi
	@echo "Running quickstart demo..."
	@./examples/quickstart.sh

install:
	@if ! command -v pip >/dev/null 2>&1; then \
		echo "Error: pip command not found. Install pip (e.g., via your Python distribution) before running 'make install'."; \
		exit 1; \
	fi
	@echo "Installing project dependencies..."
	@pip install .

dev-install:
	@if ! command -v pip >/dev/null 2>&1; then \
		echo "Error: pip command not found. Install pip (e.g., via your Python distribution) before running 'make dev-install'."; \
		exit 1; \
	fi
	@echo "Installing project dependencies with development extras..."
	@pip install .[dev]

benchmark-delta-cache:
	@if ! command -v python >/dev/null 2>&1; then \
		echo "Error: python command not found. Install Python and ensure it is on your PATH before running the benchmark."; \
		exit 1; \
	fi
	@echo "Running delta cache benchmark..."
	@python -m benchmarks.delta_cache_benchmark


report-similar-tests:
	@if ! command -v python >/dev/null 2>&1; then \
        	echo "Error: python command not found. Install Python before generating the similarity report."; \
        	exit 1; \
	fi
	@echo "Generating similarity report for pytest functions..."
	@python tools/report_similar_tests.py
	@rm -f tests/_report/*.json

report-similar-tests-sweep:
        @if ! command -v python >/dev/null 2>&1; then \
                echo "Error: python command not found. Install Python before generating the similarity report."; \
                exit 1; \
        fi
	@echo "Sweeping similarity reports for thresholds 0.90, 0.88, and 0.85..."
	@echo ""
	@echo "Running report at threshold 0.90"
	@python tools/report_similar_tests.py --threshold 0.9
	@rm -f tests/_report/*.json
	@echo ""
	@echo "Running report at threshold 0.88"
	@python tools/report_similar_tests.py --threshold 0.88
	@rm -f tests/_report/*.json
	@echo ""
	@echo "Running report at threshold 0.85"
        @python tools/report_similar_tests.py --threshold 0.85
        @rm -f tests/_report/*.json

verify-tnfr-parity:
        @if ! command -v python >/dev/null 2>&1; then \
                echo "Error: python command not found. Install Python before running the parity verification."; \
                exit 1; \
        fi
        @if [[ ! -f tests/data/synthetic_stint.csv ]]; then \
                echo "Error: tests/data/synthetic_stint.csv not found. Provide a telemetry CSV before rerunning parity verification."; \
                exit 1; \
        fi
        @echo "Running TNFR parity verification against the canonical engine..."
        @python tools/verify_tnfr_parity.py tests/data/synthetic_stint.csv --strict

