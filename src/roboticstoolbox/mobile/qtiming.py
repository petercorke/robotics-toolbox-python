import numpy as np
import heapq
import bisect
import timeit
from ansitable import ANSITable, Column
from queue import PriorityQueue

# class C:
#     def __repr__(self):
#         return 'C'
# c = C()
# q = []

# heapq.heappush(q, (1000, c))
# heapq.heappush(q, (2000, c))
# heapq.heappush(q, (3000, c))
# heapq.heappush(q, (4000, c))
# heapq.heappush(q, (5000, c))
# heapq.heappush(q, (4000, c))
# heapq.heappush(q, (3000, c))
# print(q)
# while len(q) > 0:
#     print(heapq.heappop(q))

# q = PriorityQueue()

# q.put((5, 'write code'))
# q.put((7, 'release product'))
# q.put((1_000_000, 'write spec'))
# q.put((3, 'create tests'))
# q.put((5, c))
# q.put((7, c))
# q.put((1_000_000, c))
# q.put((1_000_000, c))
# q.put((3, c))
# q.put((7, c))

# while not q.empty():
#     next_item = q.get()
#     print(next_item)

N = 1000

# setup results table
table = ANSITable(
    Column("Operation", headalign="^", colalign="<"),
    Column("Time (us)", headalign="^", fmt="{:.1f}"),
    border="thick",
)


def measure(statement):
    global table

    t = timeit.timeit(stmt=statement, setup=setup, number=N, globals=globals())

    table.row(statement, t / N * 1e6)


# ------------------------------------------------------------------------- #

# setup to run timeit
setup = """
a = list(range(1_000_000))
"""


def find(a):
    for i, x in enumerate(a):
        if x == 500_000:
            return i


measure("a.pop(0)")
measure("a.pop()")

measure("a.insert(0, 77)")
measure("a.insert(500_000, 77)")
measure("a.append(77)")

measure("a.index(500_000)")
measure("find(a)")
measure("bisect.bisect(a, 500_000)")

table.rule()
setup = """
q = [(i, 77) for i in range(1_000_000)]
"""
measure("heapq.heappush(q, (500_000, 88))")
measure("heapq.heappop(q)")
measure("[x[0] for x in q]")

table.rule()
setup = """
q = PriorityQueue()
for i in range(1_000_000):
    q.put((i, 77))
"""
measure("q.put((500_000, 88))")
measure("q.get(q)")

# pretty print the results
table.print()
