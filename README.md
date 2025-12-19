# FKAT

Foundational Kit for AI Training

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
