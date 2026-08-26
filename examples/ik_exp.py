import numpy as np
import roboticstoolbox as rtb
from ansitable import ANSITable

# Our robot and ETS
robot = rtb.models.Panda()
ets = robot.ets()

### Experiment parameters
# Number of problems to solve
problems = 10000

# Cartesion DoF priority matrix
we = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# random valid q values which will define Tep
q_rand = ets.random_q(problems)

# Our desired end-effector poses
Tep = np.zeros((problems, 4, 4))

for i in range(problems):
    Tep[i] = ets.eval(q_rand[i])

# Maximum iterations allowed in a search
ilimit = 30

# Maximum searches allowed per problem
slimit = 100

# Solution tolerance
tol = 1e-6

# Reject solutions with invalid joint limits
joint_limits = True


class IK:
    def __init__(self, name, solve, problems=problems):

        # Solver attributes
        self.name = name

        self.solve = solve

        # Solver results
        self.iterations = np.zeros(problems)
        self.searches = np.zeros(problems)
        self.residual = np.zeros(problems)
        self.success = np.zeros(problems)

        self.total_iterations = 0
        self.total_searches = 0


solvers = [
    IK(
        "Newton Raphson Pinv",
        lambda Tep: ets.ik_NR(
            Tep,
            q0=None,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=we,
            pinv=True,
            pinv_damping=0.0,
        ),
    ),
    IK(
        "Gauss Newton Pinv",
        lambda Tep: ets.ik_GN(
            Tep,
            q0=None,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=we,
            pinv=True,
            pinv_damping=0.0,
        ),
    ),
    IK(
        "LM Chan 0.1",
        lambda Tep: ets.ik_LM(
            Tep,
            q0=None,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=we,
            k=0.1,
            method="chan",
        ),
    ),
    IK(
        "LM Wampler 1e-4",
        lambda Tep: ets.ik_LM(
            Tep,
            q0=None,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=we,
            k=1e-4,
            method="wampler",
        ),
    ),
    IK(
        "LM Sugihara 0.1",
        lambda Tep: ets.ik_LM(
            Tep,
            q0=None,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=we,
            k=0.1,
            method="sugihara",
        ),
    ),
]

for i in range(problems):
    print(i + 1)

    for solver in solvers:
        sol = solver.solve(Tep[i])

        if sol.success:
            solver.success[i] = sol.success
            solver.iterations[i] = sol.iterations
            solver.searches[i] = sol.searches
            solver.residual[i] = sol.residual
            solver.total_iterations += solver.iterations[i]
            solver.total_searches += solver.searches[i]
        else:
            solver.success[i] = np.nan
            solver.iterations[i] = np.nan
            solver.searches[i] = np.nan
            solver.residual[i] = np.nan


print(f"\nNumerical Inverse Kinematics Methods Compared over {problems} problems\n")

table = ANSITable(
    "Method",
    "sLimit/iLimit",
    "Mean Steps",
    "Median Steps",
    "Infeasible",
    "Infeasible %",
    "Mean Attempts",
    # "Median Attempts",
    "Max Attempts",
    border="thin",
)

for solver in solvers:
    table.row(
        solver.name,
        f"{slimit}, {ilimit}",
        np.round(np.nanmean(solver.iterations), 2),
        np.nanmedian(solver.iterations),
        np.sum(np.isnan(solver.success)),
        np.round(np.sum(np.isnan(solver.success)) / problems * 100.0, 2),
        np.round(np.nanmean(solver.searches), 2),
        np.nanmax(solver.searches),
    )

table.print()
