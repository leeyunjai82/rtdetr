# Development and releases

Back to the [README](../README.md).

## Development

```bash
pip install -e ".[train,dev]"
pytest -q          # the parity checklist lives in tests/
ruff check .
```

## Releasing

Push a `v*` tag and GitHub Actions builds and uploads to PyPI:

```bash
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/publish.yml` uses **PyPI trusted publishing** — no API token
lives in the repo. One-time setup on PyPI (Publishing → add a pending publisher):

| Field | Value |
| --- | --- |
| PyPI project | `rtdetr` |
| Owner | `leeyunjai82` |
| Repository | `rtdetr` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

The job refuses to publish when the tag and `pyproject.toml` version disagree, so
bump the version in the same commit you tag.

