format:
	autoflake --in-place --remove-unused-variables --recursive .
	isort .
	yapf -ir .

docs:
	pdoc --html --output-dir docs src/indexing