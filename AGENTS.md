# robotics-toolbox-python — Agent Instructions

Part of the RVC ecosystem. **Read [rvc-ecosystem/AGENTS.md](https://github.com/petercorke/rvc-ecosystem/blob/main/AGENTS.md) first** — it defines shared conventions: repo ownership, math invariants, dependency boundaries, git/PR workflow, code standards, tech-debt tracking. This file only adds what's specific to this repo.

| | |
|---|---|
| PyPI package | `roboticstoolbox-python` |
| Nickname | RTB |
| Owner | Peter Corke (`petercorke`) |
| Default branch | `main` |
| Contribution model | Branch → PR; direct push to `main` at Peter's discretion |

## What the Toolbox is for

The Toolbox has three goals, in this order: **pedagogy** (code that is clear and easy to
read, and that follows the explanations in *Robotics, Vision & Control*), **functionality**
(the range of what it covers), and **performance** (speed and memory).

A faster implementation is welcome alongside a readable one (as with the C++ `ik_XX` and the
Python `ikine_XX` solvers), not instead of it. A change that gains speed by making the code
harder to understand will not be accepted. Nor will a change to an algorithm that the book
explains if the results differ from the book; a restructured implementation that gives the
same results is fine, with a comment pointing to the book section. New algorithms beyond the
book are welcome.

If your change trades clarity for speed, or changes what an algorithm does, say so in the PR
description and ask first.

## Notes specific to this repo

- Compiles `nanobind` C++ extensions via `scikit-build-core`, not Hatch — this is the
  permanent, justified exception noted in the ecosystem `AGENTS.md` build-tooling section.
- Depends on `spatialgeometry` directly (required); `swift` is an optional extra
  (`pip install roboticstoolbox-python[swift]`), not a stepping-stone to reach `spatialgeometry`.
- Uses `spatialmath` classes (`SE3`, `SO3`, `UnitQuaternion`, etc.) for all pose math — do not
  re-implement transform mathematics here.
- Tech-debt tracked as GitHub Issues labelled `tech-debt` (migrated 2026-08-09). The backlog
  was large enough that closely-related entries were clustered into parent issues with
  checklists rather than a flat 1:1 mapping — follow that pattern if filing a cluster of
  related findings rather than one issue each.
- Array type annotations use the project's own aliases, not `numpy.typing` and not an inline
  `Sequence[float] | np.ndarray`:

  ```python
  from roboticstoolbox.tools.types import ArrayLike, NDArray
  ```

  `ArrayLike` for arguments (an `ndarray`, list, tuple, range or other sequence of numbers) and
  `NDArray` for returned arrays. The shape goes in the docstring (`:type q: array_like(n)`,
  `:rtype: ndarray(6,n)`) where it adds information, not in the annotation. One definition means
  one place to change what "array-like" means.
- Docstring sections are reST, not NumPy style: a heading is ``.. rubric:: Examples`` (always
  plural), ``.. rubric:: Notes`` or ``.. rubric:: References``, never an underlined title, and
  cross-references are ``.. seealso:: :meth:`other```, never ``:seealso:`` (which Sphinx
  does not know: it renders as one more field after the parameters). A ``.. versionadded::``,
  ``.. versionchanged::`` or ``.. deprecated::`` names the release it ships in; the next
  release is the one after the version in ``pyproject.toml``.
- New code uses modern annotation syntax: `X | Y`, `X | None`, `list[X]`, `dict[K, V]`, not
  `Union`, `Optional`, `List` or `Dict`.
- Type checking: `mypy` (config in `pyproject.toml`, run it from the repo root with no arguments).
  The package is being annotated module by module, so the config lists the modules that still
  have errors; when you annotate a module, delete its entry and keep it at zero errors. Do not add
  new entries. `backends/Dynamixel` and `backends/ROS` are unmaintained sketches with no tests and
  are excluded for good: do not spend a PR annotating them.

## If you are an AI assistant preparing a contribution

- **Check for existing work before you write code.** Several agents often work on the same open
  Issue at once, producing duplicate PRs. Search for a PR for it first:
  `gh pr list --state open --search "#<issue>"`, and check the Issue's linked pull requests and
  assignee. If a PR already exists, or the Issue is assigned to someone else, do not open a
  competing PR: comment on the existing one instead. Then comment on the Issue to claim it before
  you start.
- State in the PR description whether the change alters numerical results or public behaviour.

- Read `.github/CONTRIBUTING.md`, in particular "What the checks mean". Codacy is advisory:
  Bandit is not run on `tests/` (see `.codacy.yml`), and you should not rewrite code only to
  satisfy it. The required checks are `All tests passed` and the PR title.
- Do not add files at the repository root. Documentation goes under `docs/source/` and the
  design principles are in `desiderata.md`.
- A file name mentioned in an Issue or comment may be a maintainer's private working file that
  is not in the repository. Do not create a file of that name, ask in the Issue instead.
- **Say so in the PR.** If an AI assistant wrote or substantially helped with the change,
  state that in the PR description, and which tool, so the reviewer knows what to look for. You,
  the human, remain responsible for what you submit and should have read the diff.
- **Sign the commits.** An AI-assisted commit ends with a `Co-Authored-By:` trailer naming the
  tool and model, as Claude Code does by default, and the PR description ends with the tool's
  attribution line. The repository squash-merges, so the PR description is what survives; keep
  the disclosure there as well as in the commits.
- Branch names are `type/short-description`, using the same types as the commit and PR title
  (`fix/`, `feat/`, `docs/`, `chore/`, `refactor/`, `test/`).
- Keep the PR to the one Issue or fix, and say in the description what you ran and what you
  could not run.
