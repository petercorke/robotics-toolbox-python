import sympy
import roboticstoolbox as rtb
from spatialmath.base import symbolic as sym
import time

t0 = 0


def tic():
    global t0

    t0 = time.time()


def toc(label="", p=None):
    dt = time.time() - t0
    print(f"{label:s}, {len(p.args):d} terms: elapsed time {dt:.1f}s")


puma = rtb.models.DH.Puma560(symbolic=True)
print(puma)
q = sym.symbol("q_:6")
qd = sym.symbol("qd_:6")
qdd = sym.symbol("qdd_:6")

tau = puma.rne_python(q, qd, qdd)

# create dictionaries for substitution
removetrig = {}
restoretrig = {}
squares = {}
for j in range(puma.n):
    S = sym.symbol(f"S{j:d}")
    C = sym.symbol(f"C{j:d}")
    removetrig[sym.sin(q[j])] = S
    removetrig[sym.cos(q[j])] = C
    restoretrig[S] = sym.sin(q[j])
    restoretrig[C] = sym.cos(q[j])
    squares[S**2] = 1 - C**2


def coeff_range(p):
    c = [abs(a.args[0]) for a in p.args]
    print(f"coeff range {min(c)} to {max(c)}")


def trim(p, t):
    signif_args = [a for a in p.args if abs(a.args[0]) > t]
    return p.func(*signif_args)


tic()
t1 = tau[0].expand()
toc("expand", t1)

coeff_range(t1)

t1 = trim(t1, 1e-3)
coeff_range(t1)

tic()
t1 = t1.subs(removetrig)
toc("remove trig", t1)

tic()
t1 = t1.subs(squares)
toc("squares", t1)

tic()
t1 = t1.simplify().expand()
toc("simplify + expand", t1)


tic()
t1 = t1.subs(restoretrig)
toc("restore trig", t1)

tic()
t1 = t1.simplify()
toc("simplify", t1)

print(t1)
