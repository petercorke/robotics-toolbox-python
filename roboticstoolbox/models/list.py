import roboticstoolbox.models as m
from roboticstoolbox import Robot
from ansitable import ANSITable, Column

def models(keywords=None, dof=None):
    """
    Display all robot models in summary form

    :param keywords: keywords to filter on, defaults to None
    :type keywords: tuple of str, optional
    :param dof: number of DoF to filter on, defaults to None
    :type dof: int, optional
    
    - ``models()`` displays a list of all models provided by the Toolbox.  It
      lists the name, manufacturer, model type, number of DoF, and keywords.

    - ``models(keywords=KW)`` as above, but only displays models that have a
      keyword in the tuple ``KW``.

    - ``models(dof=N)`` as above, but only display models that have ``N``
      degrees of freedom.

    The filters can be combined
    
    - ``models(keywords=KW, dof=N)`` are those models that have a keyword in
      ``KW`` and have ``N`` degrees of freedom.
    """

    table = ANSITable(
        Column("class", headalign="^", colalign="<"),
        Column("model", headalign="^", colalign="<"),
        Column("manufacturer", headalign="^", colalign="<"),
        Column("model type", headalign="^", colalign="<"),
        Column("DoF", colalign="<"),
        Column("config", colalign="<"),
        Column("keywords", headalign="^", colalign="<"),
        border="thin"
    )
    for category in ['DH', 'URDF', 'ETS']:
        group = m.__dict__[category]
        for cls in group.__dict__.values():
            if isinstance(cls, type) and issubclass(cls, Robot):
                    # we found a Robot subclass, instantiate it
                    robot = cls()
                    try:
                        config = robot.config()
                    except:
                        config = ""
                    
                    # apply filters
                    if keywords is not None:
                        if len(set(keywords) & set(robot.keywords)) == 0:
                            continue
                    if dof is not None and robot.n != dof:
                        continue

                    # add the row
                    table.row(
                        cls.__name__,
                        robot.name,
                        robot.manufacturer,
                        category,
                        robot.n,
                        config,
                        ', '.join(robot.keywords)
                    )
    table.print()

if __name__ == "__main__":
    models()
    models(keywords=('dynamics',))
    models(dof=6)
    models(keywords=('dynamics',), dof=6)