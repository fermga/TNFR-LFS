.PHONY: quickstart install dev-install

quickstart:
	./examples/quickstart.sh

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt
