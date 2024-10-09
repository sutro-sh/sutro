.PHONY: release build upload

build:
	@echo "Building the project..."
	rm -rf dist
	python -m build

install:
	@echo "Installing the project..."
	$(MAKE) build
	uv pip install $$(ls -t dist/*.whl | head -n 1) --force-reinstall

upload:
	@echo "Uploading to PyPI..."
	python -m twine upload dist/*

release:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: version parameter is required. Usage: make release X.Y.Z"; \
		exit 1; \
	fi
	@version=$(filter-out $@,$(MAKECMDGOALS)); \
	echo "Checking version $$version..."; \
	pyproject_version=$$(grep -m 1 'version = ' pyproject.toml | sed 's/version = //; s/"//g'); \
	if [ "$$version" != "$$pyproject_version" ]; then \
		echo "Error: Version mismatch. pyproject.toml version: $$pyproject_version, provided version: $$version"; \
		exit 1; \
	fi
	@echo "Version check passed. Proceeding with release $$version..."
	$(MAKE) build
	$(MAKE) upload

%:
	@: