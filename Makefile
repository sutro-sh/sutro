.PHONY: build install publish

build:
	@echo "Building the project..."
	rm -rf dist
	python3 -m build

install:
	@echo "Installing the project..."
	$(MAKE) build
	uv pip install $$(ls -t dist/*.whl | head -n 1) --force-reinstall

# Called last by the root release target; also usable to retry just PyPI.
publish:
	@set -eu; \
	  worktree_status="$$(git status --porcelain)"; \
	  test -z "$$worktree_status" || { echo 'Publish from a clean, committed checkout.' >&2; exit 1; }; \
	  release_dir="$$(mktemp -d)"; \
	  trap 'rm -rf "$$release_dir"' EXIT; \
	  uv build --out-dir "$$release_dir"; \
	  uv publish "$$release_dir"/*
