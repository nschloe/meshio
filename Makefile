VERSION=$(shell python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])")

.PHONY: default tag upload publish clean format lint

default:
	@echo "\"make publish\"?"

tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging release version v$(VERSION)..."
	# git tag v$(VERSION)
	# git push --tags
	# Always create a github "release" right after tagging so it appears on zenodo
	curl -H "Authorization: token `cat $(HOME)/.github-access-token`" -d '{"tag_name": "v$(VERSION)"}' https://api.github.com/repos/nschloe/meshio/releases

upload: clean
	# Make sure we're on the master branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	# python3 setup.py sdist bdist_wheel
	# https://stackoverflow.com/a/58756491/353337
	python3 -m pep517.build --source --binary .
	twine upload dist/*

publish: tag upload

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf *.egg-info/ build/ dist/

format:
	isort .
	black .
	blacken-docs README.md

lint:
	black --check .
	flake8 .
