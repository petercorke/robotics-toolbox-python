import roboticstoolbox as rtb
from spatialmath import *   # lgtm [py/polluting-import]
import argparse
import sys

parser = argparse.ArgumentParser(description="Puma trajectory demo")
parser.add_argument(
    '--backend',
    '-b',
    dest='backend',
    default='pyplot',
    help='choose backend: pyplot (default), swift, vpython',
    action='store')
parser.add_argument(
    '--model',
    '-m',
    dest='model',
    default='DH',
    action='store',
    help='choose model: DH (default), URDF')
args = parser.parse_args()

if args.model.lower() == 'dh':
    robot = rtb.models.DH.Puma560()
elif args.model.lower() == 'urdf':
    robot = rtb.models.URDF.Puma560()
else:
    raise ValueError('unknown model')

print(robot)

qt = rtb.tools.trajectory.jtraj(robot.qz, robot.qr, 200)

if args.backend.lower() == 'pyplot':
    if args.model.lower() != 'dh':
        print('PyPlot only supports DH models for now')
        sys.exit(1)
elif args.backend.lower() == 'vpython':
    if args.model.lower() != 'dh':
        print('VPython only supports DH models for now')
        sys.exit(1)
elif args.backend.lower() == 'swift':
    if args.model.lower() != 'urdf':
        print('Swift only supports URDF models for now')
        sys.exit(1)
else:
    raise ValueError('unknown backend')

robot.plot(qt.q, backend=args.backend)
