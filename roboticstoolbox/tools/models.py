import roboticstoolbox.models as m
from roboticstoolbox import Robot
from ansitable import ANSITable, Column

def models(category=None):
    """
    Display all robot models in summary form

    ``models()`` displays a list of all models provided by the Toolbox.  It
    lists the name, manufacturer, and number of joints.
    """

    if category is None:
        for category in ['DH', 'URDF', 'ETS']:
            models(category)
    
    else:
        print(category + ':')
        # table = ANSITable("model", "manufacturer", "DoF", "config", "keywords")
        table = ANSITable(
            Column("class", headalign="^", colalign="<"),
            Column("model", headalign="^", colalign="<"),
            Column("manufacturer", headalign="^", colalign="<"),
            Column("DoF", colalign="<"),
            Column("config", colalign="<"),
            Column("keywords", headalign="^", colalign="<"),
            border="thin"
        )
        group = m.__dict__[category]
        for cls in group.__dict__.values():
            try:
                if issubclass(cls, Robot):
                    # we found a Robot subclass, instantiate it
                    robot = cls()
                    table.row(
                        cls.__name__,
                        robot.name,
                        robot.manufacturer,
                        robot.n,
                        robot.config(),
                        ', '.join(robot.keywords)
                    )
            except:
                pass
        table.print()

if __name__ == "__main__":
    models()