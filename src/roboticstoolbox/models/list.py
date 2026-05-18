from typing import Type
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.tools import rtb_get_param
from roboticstoolbox.robot.ERobot import ERobot2
from ansitable import ANSITable, Column
import inspect

# import importlib


def list(keywords=None, dof=None, type=None, border="thin"):
    """
    Display all robot models in summary form

    :param keywords: keywords to filter on, defaults to None
    :type keywords: tuple of str, optional
    :param dof: number of DoF to filter on, defaults to None
    :type dof: int, optional
    :param type: model type "DH", "ETS", "URDF", defaults to all types
    :type type: str, optional

    - ``list()`` displays a list of all models provided by the Toolbox.  It
      lists the name, manufacturer, model type, number of DoF, and keywords.

    - ``list(type=MT)`` as above, but only displays models of type ``MT``
      where ``MT`` is one of "DH", "ETS" or "URDF".

    - ``list(keywords=KW)`` as above, but only displays models that have a
      keyword in the tuple ``KW``.

    - ``list(dof=N)`` as above, but only display models that have ``N``
      degrees of freedom.

    The filters can be combined

    - ``list(keywords=KW, dof=N)`` are those models that have a keyword in
      ``KW`` and have ``N`` degrees of freedom.
    """

    import roboticstoolbox.models as models

    # module = importlib.import_module(
    #   '.' + os.path.splitext(file)[0], package='bdsim.blocks')

    unicode = rtb_get_param("unicode")
    if not unicode:
        border = "ascii"
    def make_table(border=None):
        table = ANSITable(
            Column("class", headalign="^", colalign="<"),
            Column("name", headalign="^", colalign="<"),
            Column("manufacturer", headalign="^", colalign="<"),
            Column("type", headalign="^", colalign="<"),
            Column("DoF", colalign="<"),
            Column("dims", colalign="<"),
            Column("structure", colalign="<"),
            Column("dynamics", colalign="<"),
            Column("geometry", colalign="<"),
            Column("keywords", headalign="^", colalign="<"),
            border=border,
        )

        if type is not None:
            categories = [type]
        else:
            categories = ["DH", "URDF", "ETS"]
        for category in categories:
            # get all classes in this category
            group = models.__dict__[category]
            for cls in group.__dict__.values():
                if inspect.isclass(cls) and issubclass(cls, Robot):
                    # we found a BaseRobot subclass, instantiate it
                    try:
                        robot = cls()
                    except TypeError:
                        print(f"failed to load {cls}")
                    try:
                        structure = robot.structure
                    except Exception:  # pragma nocover
                        structure = ""

                    # apply filters
                    if keywords is not None:
                        if len(set(keywords) & set(robot.keywords)) == 0:
                            continue
                    if dof is not None and robot.n != dof:
                        continue  # pragma nocover

                    dims = 0

                    if isinstance(robot, ERobot2):
                        dims = 2
                    else:
                        dims = 3
                    # add the row
                    table.row(
                        cls.__name__,
                        robot.name,
                        robot.manufacturer,
                        category,
                        robot.n,
                        f"{dims}d",
                        structure,
                        "Y" if robot._hasdynamics else "",
                        "Y" if robot._hasgeometry else "",
                        ", ".join(robot.keywords),
                    )

        table.print()

    make_table(border=border)


if __name__ == "__main__":  # pragma nocover
    list(border='ascii')
    list(keywords=("dynamics",), border='thin')
    list(dof=6)
    list(keywords=("dynamics",), dof=6)
