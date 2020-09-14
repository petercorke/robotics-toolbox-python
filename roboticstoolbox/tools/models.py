import roboticstoolbox.models as m

def models():
    """
    Display all robot models in summary form

    ``models()`` displays a list of all models provided by the Toolbox.  It
    lists the name, manufacturer, and number of joints.
    """

    for category in ['DH', 'URDF', 'ETS']:
        print(category + ':')
        group = m.__dict__[category]

        for cls in group.__dict__.values():
            # TODO should check that cls issubclass of Robot superclass (when there is one)
            try:
                robot = cls()
            except:
                continue

            s = robot.name
            if robot.manufacturer is not None:
                s += ' (' + robot.manufacturer + ')'
            print(f"  {s:40s} {robot.n:d} dof")

        
models()