"""
Python ReedShepp Planner
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""

from numpy import disp
from scipy import integrate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *


class ReedsShepp:
    def __init__(self, q0, qf, max_curve, d1):
        self._max_c = max_curve

        # The word describing the shortest path
        self._words = generate_path(q0, qf, self._max_c)

        if not any(self._words):
            Error("No path.")

        # Find the shortest path
        L = [word["L"] for word in self._words]
        k = np.argmin(L)
        self._best_path = self._words[k]
        self._best_path = generate_trajectories(self._best_path, self._max_c, d1, q0)

        disp("Words")
        disp(self._words)
        disp("Best Path")
        disp(self._best_path)

    @property
    def best_path(self):
        return self._best_path

    @property
    def words(self):
        return self._words

    @property
    def max_c(self):
        return self._max_c

    def path(self):
        return [np.transpose(np.all(self._best_path.trajectory))]

    def show(self):
        for w in self._words:
            print("%s (%g): [%g %g %g]\n", w.word, w.L, w.lengths)

    # This may be re-written in the future to utilise internal/library graphics.
    def plot(self):
        # Declare variables to include them in scope.
        x = None
        y = None
        t = None
        r = None
        c = None

        word = self._best_path
        for i in range(3):
            if word["dir"][i] > 0:
                color = "b"
            else:
                color = "r"

            if i == 0:
                x = word["traj"][i][:, 0]
                y = word["traj"][i][:, 1]
            else:
                x = word["traj"][i][:, 0]
                y = word["traj"][i][:, 1]

            plt.plot(x, y, color, linewidth=2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.title("Reeds-Shepp Path")
        plt.show()


def generate_trajectories(words, max_c, d, q0):
    p0 = q0
    out = words
    out["traj"] = [[], [], []]
    out["dir"] = [[], [], []]

    for i in range(3):
        m = words["word"][i]
        l = words["lengths"][i]
        x = np.arange(0, np.abs(l), d)

        if x[-1] != np.abs(l):
            x = np.append(x, np.abs(l))

        p = pathseg(x, np.sign(l), m, max_c, p0)

        if i == 0:
            out["traj"][i] = p
        else:
            out["traj"][i] = p
            np.delete(out["traj"][i], 0)

        out["dir"][i] = np.sign(l)
        p0 = p[-1]

    return out


def pathseg(l, dir, m, max_c, p0):
    f = None
    q = []
    q0 = p0
    # PEP 8 doesn't like this but it's the nicest way to define a function we'll use for the ODE
    if m == "s":
        f = lambda f_t, f_q: (
            dir * np.transpose(np.array([np.cos(f_q[2]), np.sin(f_q[2]), 0]))
        )
    elif m == "l" or m == "r":
        f = lambda f_t, f_q: (
            dir * np.transpose(np.array([np.cos(f_q[2]), np.sin(f_q[2]), dir * max_c]))
        )

    results = integrate.solve_ivp(f, [l[0], l[-1]], q0, t_eval=l, method="RK45")
    q = np.transpose(results.y)
    return q


def generate_path(q0, q1, max_c):
    q0 = q0
    q1 = q1
    dq = q1 - q0
    dth = dq[2]

    xy = np.transpose(rot2(q0[2])) * dq[:2] * max_c
    xy = xy[1]
    x = xy[0]
    y = xy[1]

    words = []
    words = scs(x, y, dth, words)
    words = csc(x, y, dth, words)
    words = ccc(x, y, dth, words)

    for i in range(len(words)):
        words[i]["lengths"] = words[i]["lengths"] / max_c
        words[i]["L"] = words[i]["L"] / max_c

    return words


def scs(x, y, phi, words):
    words = sls([x, y, phi], 1, "sls", words)
    words = sls([x, -y, -phi], 1, "srs", words)

    return words


def ccc(x, y, phi, words):
    words = lrl([x, y, phi], 1, "lrl", words)
    words = lrl([-x, y, -phi], -1, "lrl", words)
    words = lrl([x, -y, -phi], 1, "rlr", words)
    words = lrl([-x, -y, phi], -1, "rlr", words)

    # Backwards
    xb = x * np.cos(phi) + y * np.sin(phi)
    yb = x * np.sin(phi) - y * np.cos(phi)

    flip = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]  # flip u and v

    words = lrl([xb, yb, phi], flip, "lrl", words)
    words = lrl([-xb, yb, -phi], np.negative(flip), "lrl", words)
    words = lrl([xb, -yb, -phi], flip, "rlr", words)
    words = lrl([-xb, -yb, phi], np.negative(flip), "rlr", words)

    return words


def csc(x, y, phi, words):
    words = lsl([x, y, phi], 1, "lsl", words)
    words = lsl([-x, y, -phi], -1, "lsl", words)
    words = lsl([x, -y, -phi], 1, "rsr", words)
    words = lsl([-x, -y, phi], -1, "rsr", words)
    words = lsr([x, y, phi], 1, "lsr", words)
    words = lsr([-x, y, -phi], -1, "lsr", words)
    words = lsr([x, -y, -phi], 1, "rsl", words)
    words = lsr([-x, -y, phi], -1, "rsl", words)

    return words


def sls(q, sign, word, words):
    x = q[0]
    y = q[1]
    phi = np.mod(q[2], 2 * np.pi)

    if y > 0 and 0.0 < phi < np.pi * 0.99:
        xd = -y / np.tan(phi) + x
        t = xd - np.tan(phi / 2.0)
        u = phi
        v = np.norm([(x - xd), y]) - np.tan(phi / 2.0)
        return add_path(words, sign * np.array([t, u, v]), word)
    elif y < 0 and 0.0 < phi < np.pi * 0.99:
        xd = -y / np.tan(phi) + x
        t = xd - np.tan(phi / 2.0)
        u = phi
        v = -np.norm([(x - xd), y]) - np.tan(phi / 2.0)
        return add_path(words, sign * np.array([t, u, v]), word)
    else:
        return words


def lsl(q, sign, word, words):
    x = q[0]
    y = q[1]
    phi = np.mod(q[2], 2 * np.pi)
    [t, u] = cart2pol(x - np.sin(phi), y - 1.0 + np.cos(phi))
    if t >= 0:
        v = short_angdiff(phi - t)
        if v >= 0:
            return add_path(words, sign * np.array([t, u, v]), word)

    return words


def lrl(q, sign, word, words):
    x = q[0]
    y = q[1]
    phi = np.mod(q[2], 2 * np.pi)
    [t1, u1] = cart2pol(x - np.sin(phi), y - 1.0 + np.cos(phi))

    if u1 <= 4.0:
        u = -2.0 * (np.arcsin(0.25 * u1))
        t = short_angdiff(t1 + 0.5 * u + np.pi)
        v = short_angdiff(phi - t + u)
        if t >= 0.0 and u <= 0.0:
            return add_path(words, np.array([t, u, v]) * sign, word)

    return words


def lsr(q, sign, word, words):
    x = q[0]
    y = q[1]
    phi = np.mod(q[2], 2 * np.pi)

    [t1, u1] = cart2pol(x + np.sin(phi), y - 1.0 - np.cos(phi))
    u1 = np.square(u1)

    if u1.all() >= 4.0:
        u = np.sqrt(u1 - 4.0)
        theta = np.arctan(2.0, u)
        t = short_angdiff(t1 + theta)
        v = short_angdiff(t - phi)

        if t.all() >= 0.0 and v >= 0.0:
            return add_path(words, sign * np.array([t, u, v]), word)

    return words


def add_path(words, lengths, c_types):
    # Create a struct to represent this segment
    word = {"word": c_types, "lengths": lengths, "L": None}

    # Check same path exist
    for p in words:
        if p["word"] == word["word"]:
            if (np.sum(p["lengths"]) - np.sum(word["lengths"])) <= 0.01:
                return words  # Don't insert path

    word["L"] = np.sum(np.abs(lengths))

    # long enough to add?
    if word["L"] >= 0.01:
        words.append(word)
        return words


# Peter's angdiff doesn't function quite like Matlab's, so this'll work with just argument a
def short_angdiff(matrix):
    if isinstance(matrix, np.float64):
        matrix = [matrix, 2 * np.pi]
    ad = angdiff(matrix[0], matrix[1])
    return ad


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi, rho


class Error(Exception):
    """Base class for other exceptions"""

    pass
