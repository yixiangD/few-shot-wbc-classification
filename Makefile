dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

check:
	isort -c src/
	black src/ --check
	flake8 src/

format:
	isort src/
	black src/
	flake8 src/

.PHONY: dev check
