.PHONY: quickstart install dev-install benchmark-delta-cache

quickstart:
	./examples/quickstart.sh

install:
        pip install .

dev-install:
        pip install .[dev]

benchmark-delta-cache:
	python -m benchmarks.delta_cache_benchmark
