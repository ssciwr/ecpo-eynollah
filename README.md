# Welcome to ecpo-eynollah

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/ecpo-eynollah/ci.yml?branch=main)](https://github.com/ssciwr/ecpo-eynollah/actions/workflows/ci.yml)

## Installation

The Python package `ecpo_eynollah` can be installed from PyPI:

```
python -m pip install ecpo_eynollah
```

## Development installation

If you want to contribute to the development of `ecpo_eynollah`, we recommend
the following editable installation from this repository:

```
git clone git@github.com:ssciwr/ecpo-eynollah.git
cd ecpo-eynollah
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
