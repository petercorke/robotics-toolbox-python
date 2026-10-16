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

- Name branches `type/short-description`, with the same types as the PR title (`fix/`, `feat/`, `docs/`, `chore/`, `refactor/`, `test/`).
- Target branch is `main`.

## AI-assisted contributions

AI coding assistants are welcome, but please say so in the PR description (and which tool),
so the reviewer knows what to look for. Also keep the `Co-Authored-By:` trailer on the commits
(most tools add it), since the repository squash-merges and the PR description is what is kept.
You remain responsible for what you submit, so read
the diff yourself and run the tests before asking for review.

## What the checks mean

Every PR runs a set of automatic checks. Not all of them are equal:

- **Required:** `All tests passed` (the test matrix on Linux, macOS and Windows, the `mypy`
  type check and the docs build) and the PR title check. These need to be green to merge.
- **Advisory, not required:** the Codacy static analysis. It is known to flag `assert`
  statements in tests as "high" security issues (bandit B101). Please do not rewrite tests
  or code to satisfy it, a maintainer will look at anything that matters.
- **First pull request:** checks do not start until a maintainer approves the workflow run,
  which GitHub requires for first-time contributors. A PR with no checks yet is normal, not
  a failure.
- **A flaky timeout** on one Windows or macOS job is common. Say so in a comment and a
  maintainer will re-run it.

## Where things go

- Design notes, plans and write-ups do not go in the repository root. User-facing
  documentation goes in `docs/source/`, and the project's design principles are in
  `desiderata.md`.
- Please do not add test files, data files or notebooks specific to your own project.

## License

By contributing, you agree that your contributions will be licensed under this project's [MIT License](LICENSE).
