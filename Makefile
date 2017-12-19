VERSION=$(shell python -c "import meshio; print(meshio.__version__)")

# Make sure we're on the master branch
ifneq "$(shell git rev-parse --abbrev-ref HEAD)" "master"
$(error Not on master branch)
endif

default:
	@echo "\"make publish\"?"

README.rst: README.md
	cat README.md | sed -e 's_<img src="\([^"]*\)" width="\([^"]*\)">_![](\1){width="\2"}_g' -e 's_<p[^>]*>__g' -e 's_</p>__g' > /tmp/README.md
	pandoc /tmp/README.md -o README.rst
	python setup.py check -r -s || exit 1

upload: setup.py README.rst
	rm -f dist/*
	python setup.py bdist_wheel --universal
	gpg --detach-sign -a dist/*
	twine upload dist/*

tag:
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

publish: tag upload

clean:
	rm -f README.rst
