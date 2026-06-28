# Technical Debt

## Link deepcopy drops collision geometry (coal pickling limitation)

`Link.__deepcopy__` silently drops coal `CollisionObject` instances because the
coal library does not support pickling. The workaround warns at runtime when
shapes are lost.

The root cause is that a fresh `DHLink` can reach the coal objects of an
unrelated URDF robot through shared class-level state in `Link` or `Robot`
(exact attribute TBD). This means:

- Copied DH links that happen to run after URDF robot tests lose nothing (DH
  links have no collision shapes), but the warning path is never exercised in
  isolation.
- If the shared reference is ever followed for other purposes (e.g. iteration,
  serialisation) it could cause similar failures or unexpected aliasing.

**Proper fix (Option 2):** obtain the full `deepcopy` traceback with
`--tb=long` when running `test_ERobot.py` followed by
`test_Link.py::TestDHLink::test_copy`, trace which attribute chain connects the
fresh `DHLink` to a coal object, and remove or weak-ref that shared state.
