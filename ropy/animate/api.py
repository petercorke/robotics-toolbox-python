import zerorpc
import ropy as rp

panda = rp.PandaMDH()
panda.q = panda.qr

l = []
for i in range(panda.n):
    # m = min(0, i-1)
    # l.append(panda.A([m, i]).A.tolist())
    # lib = panda.A(i).A
    # lie = panda.links[i].A(panda.q[i]).A
    # l.append(lib.tolist())
    # l.append(lie.tolist())
    li = [panda.links[i].sigma, panda.links[i].mdh, panda.links[i].theta, panda.links[i].d, panda.links[i].a, panda.links[i].alpha]
    l.append(li)

# l1 = panda.A([0, 1])



c = zerorpc.Client()
c.connect("tcp://127.0.0.1:4242")
print(c.hello(l))
