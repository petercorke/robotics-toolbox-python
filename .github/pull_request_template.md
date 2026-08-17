Thanks for contributing to RTB!

## Summary

<!-- What does this PR do, and why? -->

## Related issue

<!-- Fixes #123 / Closes #123 — if applicable -->

## Checklist

Only the first item below is checked automatically — the rest are a self-check for you before requesting review, nothing currently verifies them for you.

- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (`type: description`) — checked automatically, see the "Check PR title" status
- [ ] Tests pass locally (`pytest`)
- [ ] Added/updated tests for this change, if applicable
- [ ] New/changed code is type-hinted with modern syntax (`X | Y`, `list[X]`, not `Union`/`Optional`/`List`)
- [ ] Docstrings updated (reST style: `:param:`, `:returns:`; type hints in the signature cover types now, `:type:`/`:rtype:` are rarely needed)
- [ ] PR is as small/focused as practical — if it tackles several unrelated things, consider splitting it so each can be reviewed and accepted independently
- [ ] No test files, data files, or notebooks specific to your own project — PyPI has strict package size limits, and RTB is already split into a toolbox and a data package. Notebooks, if of general interest, should have output cleared before committing.

<!-- Target branch is `main`. -->
