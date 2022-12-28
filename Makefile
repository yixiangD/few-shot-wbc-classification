dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

check:
	isort -c src/
	black src/ --check
	flake8 src/
	isort -c few_shot_wbc/
	black few_shot_wbc/ --check
	flake8 few_shot_wbc/

format:
	isort src/
	black src/
	flake8 src/
	isort few_shot_wbc/
	black few_shot_wbc/
	flake8 few_shot_wbc/

.PHONY: dev check
