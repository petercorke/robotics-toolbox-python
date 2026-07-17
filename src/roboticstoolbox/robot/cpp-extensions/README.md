# Fast recursive Newton-Euler (frne)

## Synopsis

nanobind C++/C extension that accelerates `DHRobot.rne()`: recursive
Newton-Euler inverse dynamics. These files were developed for a "MEX
version" of robot dynamics in the MATLAB version of the Robotics Toolbox
— `ne.c` is virtually unchanged.

This is one of two such extensions in the codebase — the other, **fknm**
(`_fknm_c`, ETS forward kinematics/Jacobians/Hessians/IK), lives at
`ets/cpp-extensions/` since it's an ETS concern, not a dynamics one. See
that directory's README for the architecture patterns shared by both
extensions (build/install mechanics, the Python facade + pure-Python
fallback pattern, lazy serialization into the C++ mirror object).

It's optional: `robot/frne.py` is a pure-Python facade that tries to
import the compiled extension and transparently falls back to an
equivalent pure-Python implementation when it's unavailable.

# Files

| File | Purpose |
| ---- | ------- |
| frne_nb.cpp | nanobind glue for `_frne_c` — binds `init`/`frne`/`delete` |
| frne.h | struct/function declarations shared with `ne.c` |
| ne.c | implementation of the RNE algorithm |
| vmath.c | Simple vector/matrix library |
| vmath.h | Simple vector/matrix library |
