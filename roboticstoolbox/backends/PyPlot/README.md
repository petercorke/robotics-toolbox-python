Backend methods

- launch, creates the axes
- close, close the axes
- add, a robot, ellipse/oid, shape
- remove, a robot, ellipse/oid, shape
- step, update all shapes


## add

```
add(shape, [shape specific options])

    if robot:
        # create a mirror object that has a reference to the user's object
        # call .draw() on it
    if ellipse:
        # add it to list
        # call .draw() on it
```

robot object options:

- readonly,
- display,
- jointaxes, show joint axes
- jointlabels, show joint labels
- eeframe, show ee frame
- shadow, show shadow
- name, show name
- options, detailed appearance options for robot, shadow, labels, eeframe

`.draw()`
- onetime, init the line segments
- compute fkine_all
- update line segments, shadows, ee frame


ellipse object options:

- robot, robot it is attached to
- q
- etype, ellipse type: v or f
- opt
- centre
- scale

Attributes:
- ell, the MPL handle

`.draw()` 
- remove old ellipse
- compute Jacobian to get ellipse matrix
- compute fkine to get ellipse centre
- draw new ellipse

## step

```
step(dt)

    # update state by control mode p/v/a
    # call .draw() on all robots in list
    # call .draw() on all ellipses in list
```

