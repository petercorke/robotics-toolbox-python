def plot(self, jointconfig, unit='rad'):
    """
    Creates a 3D plot of the robot in your web browser
    :param jointconfig: takes an array or list of joint angles
    :param unit: unit of angles. radians if not defined
    :return: a vpython robot object.
    """
    if type(jointconfig) == list:
        jointconfig = argcheck.getvector(jointconfig)
    if unit == 'deg':
        jointconfig = jointconfig * pi / 180
    if jointconfig.size == self.length:
        poses = self.fkine(jointconfig, unit, alltout=True)
        t = list(range(len(poses)))
        for i in range(len(poses)):
            t[i] = poses[i].t
        # add the base
        t.insert(0, SE3(self.base).t)
        grjoints = list(range(len(t) - 1))

        canvas_grid = init_canvas()

        for i in range(1, len(t)):
            grjoints[i - 1] = RotationalJoint(vector(t[i - 1][0], t[i - 1][1], t[i - 1][2]),
                                              vector(t[i][0], t[i][1], t[i][2]))
        print(grjoints)
        x = GraphicalRobot(grjoints)

        return x