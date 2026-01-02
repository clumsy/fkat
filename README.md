# FKAT

[![Documentation](https://img.shields.io/badge/docs-gh--pages-blue)](https://amzn.github.io/fkat/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Foundational Kit for AI Training

## Documentation

📚 **[Read the full documentation](https://amzn.github.io/fkat/)**

## Dependencies

This project depends on third-party open source packages that are installed via PyPI.

Key dependencies include:
- PyTorch (BSD-3-Clause)
- Lightning (Apache-2.0)
- Transformers (Apache-2.0)
- Hydra (MIT)
- MLflow (Apache-2.0)
- AWS SDK for Python / Boto3 (Apache-2.0)
- PyArrow (Apache-2.0)

For a complete list of dependencies and their licenses, see `pyproject.toml` and run `pip-licenses` after installation.

## Setup

```bash
pip install hatch
hatch env create
```

## Development

```bash
hatch run test:test
hatch run lint:check
```

## Documentation

Docs are automatically built and deployed to GitHub Pages on push to main/mainline.

Build locally:
```bash
hatch run docs:build
hatch run docs:serve
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
