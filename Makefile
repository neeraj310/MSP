format:
	autoflake -i **/**/*.py
	isort **/**/*.py
	yapf -i **/**/*.py
