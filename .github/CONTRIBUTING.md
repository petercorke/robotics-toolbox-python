# Contributing to the Robotics Toolbox for Python

Thanks for your interest in contributing! We welcome bug reports, fixes, new features, and documentation improvements.

## Reporting issues

Please use the issue template and include:

- Your operating system, Python version, and roboticstoolbox version
- A short, self-contained code example that reproduces the problem

## Looking for somewhere to start?

Issues labelled [`good first issue`](https://github.com/petercorke/robotics-toolbox-python/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or [`help wanted`](https://github.com/petercorke/robotics-toolbox-python/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) are a good place to start.

## Contributing code

- Keep pull requests scoped to a single feature or fix. If you have several unrelated changes, open separate PRs so each can be reviewed and merged independently.
- For API changes, propose the change in [Discussions](https://github.com/petercorke/robotics-toolbox-python/discussions) before opening a PR.
- PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) (`type: description`) — checked automatically on the PR.
- Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/); configuration is in `pyproject.toml` under `[tool.ruff]`.
- New or changed code should be type-hinted using modern syntax (`X | Y`, `X | None`, `list[X]`, `dict[K, V]` — not `Union`, `Optional`, `List`, `Dict`).
- Docstrings use reST style (`:param:`, `:returns:`). Type hints in the function signature already cover types, so `:type:`/`:rtype:` are rarely needed.
- Any code change should be covered by tests and must not break existing ones. Tests live in `tests/`. Install the dev dependencies with `pip install -e '.[dev]'` (add `,docs` too if you're touching documentation), then run:

  ```
  pytest tests/ --ignore=tests/test_blocks.py --timeout=50 --timeout_method=thread -q
  ```

- Target branch is `main`.

## License

By contributing, you agree that your contributions will be licensed under this project's [MIT License](LICENSE).
