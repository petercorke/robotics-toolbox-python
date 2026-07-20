import warnings
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.tools import rtb_get_param
from roboticstoolbox.robot.ERobot import ERobot2
from roboticstoolbox.models.URDF.URDFRobot import _rd_link
from ansitable import ANSITable, Column
import inspect

# import importlib


def catalog(keywords=None, dof=None, mtype=None, border="thin"):
    """
    Display all robot models in summary form

    :param keywords: keywords to filter on, defaults to None
    :type keywords: tuple of str, optional
    :param dof: number of DoF to filter on, defaults to None
    :type dof: int, optional
    :param mtype: model type "DH", "ETS", "URDF", defaults to all types
    :type mtype: str, optional

    - ``catalog()`` displays a list of all models provided by the Toolbox.
      It lists the name, manufacturer, model type, number of DoF, and
      keywords. It also lists models importable by name from
      `robot_descriptions <https://github.com/robot-descriptions/robot_descriptions.py>`_
      but not otherwise wrapped by the Toolbox.

    - ``catalog(mtype=MT)`` as above, but only displays Toolbox models of
      type ``MT`` where ``MT`` is one of "DH", "ETS" or "URDF", and omits
      the robot_descriptions listing.

    - ``catalog(keywords=KW)`` as above, but only displays models that have
      a keyword in the tuple ``KW``.

    - ``catalog(dof=N)`` as above, but only display models that have ``N``
      degrees of freedom.

    The filters can be combined

    - ``catalog(keywords=KW, dof=N)`` are those models that have a keyword
      in ``KW`` and have ``N`` degrees of freedom.
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
            Column("structure", colalign="<", width=16),
            Column("dynamics", colalign="<"),
            Column("geometry", colalign="<"),
            Column("keywords", headalign="^", colalign="<"),
            border=border,
        )

        if mtype is not None:
            categories = [mtype]
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
                    except Exception:
                        print(f"failed to load {cls}")
                        continue
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

    def make_rd_table(border=None):
        from robot_descriptions import DESCRIPTIONS
        from robot_descriptions._descriptions import Format

        table = ANSITable(
            Column("name", headalign="^", colalign="<"),
            Column("robot", headalign="^", colalign="<"),
            Column("manufacturer", headalign="^", colalign="<"),
            Column("DoF", colalign="<"),
            Column("keywords", headalign="^", colalign="<"),
            border=border,
        )

        for key, description in sorted(DESCRIPTIONS.items()):
            if Format.URDF not in description.formats:
                continue

            name = key.removesuffix("_description").removesuffix("_official")
            tags = sorted(description.tags)

            if keywords is not None:
                if len(set(keywords) & set(tags)) == 0:
                    continue
            if dof is not None and description.dof != dof:
                continue

            table.row(
                name,
                description.robot,
                description.maker,
                description.dof if description.dof is not None else "",
                ", ".join(tags),
            )

        print(f"\nImportable from {_rd_link()}\n")
        table.print()

    make_table(border=border)
    if mtype is None:
        make_rd_table(border=border)


def list(keywords=None, dof=None, mtype=None, border="thin"):  # pragma nocover
    """
    Display all robot models in summary form (deprecated)

    :deprecated: 1.4.0
        ``list`` shadows the builtin and will be removed in a future
        release. Use :func:`catalog` instead.
    """
    warnings.warn(
        "models.list() is deprecated and will be removed in a future "
        "release, use models.catalog() instead",
        FutureWarning,
        stacklevel=2,
    )
    return catalog(keywords=keywords, dof=dof, mtype=mtype, border=border)


if __name__ == "__main__":  # pragma nocover
    catalog(border="ascii")
    catalog(keywords=("dynamics",), border="thin")
    catalog(dof=6)
    catalog(keywords=("dynamics",), dof=6)
