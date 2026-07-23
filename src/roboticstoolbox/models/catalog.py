import warnings
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.tools import rtb_get_param
from roboticstoolbox.robot.ERobot import ERobot2
from roboticstoolbox.models.URDF.URDFRobot import _rd_link
from ansitable import ANSITable, Column
import inspect

# import importlib


def catalog(keywords=None, dof=None, mtype=None, sorton=None, border="thin"):
    """
    Display all robot models in summary form

    :param keywords: keywords to filter on, defaults to None
    :type keywords: tuple of str, optional
    :param dof: number of DoF to filter on, defaults to None
    :type dof: int, optional
    :param mtype: filter on model type: "DH", "ETS", "URDF", defaults to all types
    :type mtype: str, optional
    :param sorton: column to sort rows on: "name", "manufacturer" or "dof",
        case insensitive, defaults to None (construction order)
    :type sorton: str, optional

    - ``catalog()`` displays a list of all models provided by the Toolbox.
      It lists the name, manufacturer, model type, number of DoF, and
      keywords. It also lists models available from
      `robot_descriptions <https://github.com/robot-descriptions/robot_descriptions.py>`_
      but not otherwise wrapped by the Toolbox: "robot_descriptions name"
      is the string ``robot_descriptions`` itself calls the *name* (its own
      ``python -m robot_descriptions pull NAME`` CLI) -- pass it to
      :class:`~roboticstoolbox.models.URDF.URDFRobot.URDFRobot` to load that
      model, e.g. ``URDFRobot("ur5")``.

    - ``catalog(mtype=MT)`` as above, but only displays Toolbox models of
      type ``MT`` where ``MT`` is one of "DH", "ETS" or "URDF". The
      robot_descriptions listing (all URDF) is shown for ``MT="URDF"`` or
      unfiltered, and omitted for ``MT="DH"``/``MT="ETS"``.

    - ``catalog(keywords=KW)`` as above, but only displays models that have
      a keyword in the tuple ``KW``.

    - ``catalog(dof=N)`` as above, but only display models that have ``N``
      degrees of freedom.

    - ``catalog(sorton=S)`` as above, but rows are sorted by column ``S``.
      ``name`` sorts by the robot's name, ``manufacturer`` sorts by the robot's
      manufacturer, and ``dof`` sorts by the robot's number of degrees of freedom. Sorting is case insensitive.
      Applies to both tables when both are shown. Rows with no ``DoF``
      value (common in the robot_descriptions table) sort first.

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

    # sorton maps to (column-in-main-table, column-in-rd-table, key). The two
    # tables spell the "robot's own name" column differently -- "name" here,
    # "robot name" in the robot_descriptions table, which also has a second,
    # distinct "robot_descriptions name" column (see make_rd_table) -- so
    # sorton="name" resolves to a different literal header per table.
    sort_col_main = None
    sort_col_rd = None
    sort_key = None
    if sorton is not None:
        sort_map = {
            "name": ("robot name", "robot name", None),
            "manufacturer": ("manufacturer", "manufacturer", None),
            "dof": ("DoF", "DoF", lambda s: int(s) if s.strip() else -1),
        }
        try:
            sort_col_main, sort_col_rd, sort_key = sort_map[sorton.lower()]
        except KeyError:
            raise ValueError(
                f"sorton must be one of {sorted(sort_map)}, got {sorton!r}"
            )

    def make_table(border=None):
        table = ANSITable(
            Column("class", headalign="^", colalign="<"),
            Column("robot name", headalign="^", colalign="<"),
            Column("manufacturer", headalign="^", colalign="<"),
            Column("model type", headalign="^", colalign="<"),
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

        if sort_col_main is not None:
            table.sort(sort_col_main, key=sort_key)
        table.print()

    def make_rd_table(border=None):
        from robot_descriptions import DESCRIPTIONS
        from robot_descriptions._descriptions import Format

        table = ANSITable(
            Column("robot_descriptions name", headalign="^", colalign="<"),
            Column("robot name", headalign="^", colalign="<"),
            Column("manufacturer", headalign="^", colalign="<"),
            Column("model type", headalign="^", colalign="<"),
            Column("DoF", colalign="<"),
            Column("keywords", headalign="^", colalign="<"),
            border=border,
        )

        for key, description in sorted(DESCRIPTIONS.items()):
            if Format.URDF not in description.formats:
                continue

            rd_name = key.removesuffix("_description").removesuffix("_official")
            tags = sorted(description.tags)

            if keywords is not None:
                if len(set(keywords) & set(tags)) == 0:
                    continue
            if dof is not None and description.dof != dof:
                continue

            table.row(
                rd_name,
                description.robot,
                description.maker,
                "URDF",
                description.dof if description.dof is not None else "",
                ", ".join(tags),
            )

        print(f"\nImportable from {_rd_link()}:\n")
        if sort_col_rd is not None:
            table.sort(sort_col_rd, key=sort_key)
        table.print()

    make_table(border=border)
    if mtype is None or mtype == "URDF":
        make_rd_table(border=border)


def list(  # pragma nocover
    keywords=None, dof=None, mtype=None, sorton=None, border="thin"
):
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
    return catalog(
        keywords=keywords, dof=dof, mtype=mtype, sorton=sorton, border=border
    )


if __name__ == "__main__":  # pragma nocover
    catalog(border="ascii")
    catalog(keywords=("dynamics",), border="thin")
    catalog(dof=6)
    catalog(keywords=("dynamics",), dof=6)
