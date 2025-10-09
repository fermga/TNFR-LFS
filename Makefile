.PHONY: quickstart install dev-install benchmark-delta-cache

quickstart:
	./examples/quickstart.sh

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt

benchmark-delta-cache:
	python -m benchmarks.delta_cache_benchmark
