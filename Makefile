SHELL := /bin/bash

.PHONY: quickstart install dev-install benchmark-delta-cache
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
