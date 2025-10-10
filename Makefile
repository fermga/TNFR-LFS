.PHONY: quickstart install dev-install benchmark-delta-cache

quickstart:
	./examples/quickstart.sh

install:
	python -m pip install .

dev-install:
	python -m pip install ".[dev]"

benchmark-delta-cache:
	python -m benchmarks.delta_cache_benchmark
