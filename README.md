# materialized-client

The official Python client for Materialized Intelligence. See [docs.materialized.dev](https://docs.materialized.dev/) for more information.

## Installing Locally (to test changes during development)

Run `make install` from the root directory. This should remove the old builds and reinstall the package in your environment with the latest. You can run `uv pip list` to ensure the package is pointing at the local files instead of the PyPI package.

## Creating releases

Make sure you increment the version appropriately in `pyproject.toml`. Generally speaking we'll do patch versions for small tweaks, minor versions for large additions or changes to behavior, and probably do major releases once it makes sense. Since we're still in beta and `0.x.x` releases, its probably okay to add backwards-incompatible changes to minor releases, but we want to avoid this if possible. 

To create a release, run: 

`make release <version>` with `<version>` formatted like `0.1.1`

It'll prompt you for an API key to PyPI, which you must have for it to work. 

We also have a test PyPI account which you can use to test creating releases before pushing to the actual PyPI hub. I believe you can only create **one** release per version number, so it may be worth testing if you're paranoid about getting it right.

Also make sure to update the docs and increment the docs version number to match the new release. Keeping these consistent will provide a better user experience. 