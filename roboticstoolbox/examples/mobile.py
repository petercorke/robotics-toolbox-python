# chapter 5 code as per published


# ## braitenberg
# sl_braitenberg
# sim('sl_braitenberg')

## simple automata

from scipy.io import loadmat
from roboticstoolbox import Bug2, DXform

vars = loadmat("/Users/corkep/code/robotics-toolbox-python/data/house.mat", squeeze_me=True, struct_as_record=False)
house = vars['house']
place = vars['place']
# bug = Bug2(house)
# p = bug.query(place.br3, place.kitchen, animate=True)

vars = loadmat("/Users/corkep/code/robotics-toolbox-python/data/map1.mat", squeeze_me=True, struct_as_record=False)
map = vars['map']
bug = Bug2(map)
# # bug.plot()
# p = bug.query([20, 10], [50, 35], animate=True)
# print(p)
# p = bug.query(place.br3, place.kitchen)
# about(p)
# p = bug.query([], place.kitchen) 

# bug = Bug2(house), inflate=7)
# p = bug.query(place.br3, place.kitchen, animate=False)

# bug.plot(p, 'inflated')

# ## making a map (sidebar)
# map = zeros(100, 100)
# map(40:50,20:80) = 1
# map = makemap(100)

## map based planning

dx = DXform(house)
dx.plan(place.kitchen)

dx.plot()

p = dx.query(place.br3) #, animate=True)
print(p)

dx.plot(path=p, block=True)

# p = dx.query(place.br3)

# dx.plot(p)

# numrows(p)

# dx.plan(goal, 'animate')

# dx.plot3d(p)

# # inflation
# dx = DXform(house, 'inflate', 5)
# dx.plan(place.kitchen)
# p = dx.query(place.br3)
# dx.plot(p)


# ## Navigation class (sidebar)
# nav = MyNavClass(world)
# nav.plan()
# nav.plan(goal)

# p = nav.query(start, goal)
# p = nav.query(start)
# nav.plot()
# nav.plot(p)

# ## D* planner
# ds = Dstar(house)
# c = ds.costmap()
# ds.plan(place.kitchen)
# ds.niter
# ds.query(place.br3)

# ds.modify_cost( [300,325 115,125], 5 )

# ds.plan()
# ds.niter
# ds.query(place.br3)


# ## Roadmap methods

# free = 1 - house
# free(1,:) = 0 free(100,:) = 0
# free(:,1) = 0 free(:,100) = 0
# skeleton = ithin(free)


# ## 5.2 PRM
# prm = PRM(house)
# randinit
# prm.plan('npoints', 150)
# prm
# prm.plot()

# p = prm.path(place.br3, place.kitchen)
# about p

# ## random number sidebar

# rand
# rand
# rand
# randinit
# rand
# rand

# ## 5.3 Lattice planner

# lp = Lattice()
# lp.plan('iterations', 2)
# lp.plot()

# lp.plan('iterations', 8)
# lp.plot()

# lp.query( [1 2 pi/2], [2 -2 0] )

# lp.plot()

# p = lp.query( [1 2 pi/2], [2 -2 0] )
# about p

# lp.plan('cost', [1 10 10])
# lp.query(start, goal)
# lp.plot()

# load road
# lp = Lattice(road, 'grid', 5, 'root', [50 50 0])
# lp.plan()
# lp.query([30 45 0], [50 20 0])


# ## RRT planner
# car = Bicycle('steermax', 0.5)
# rrt = RRT(car, 'npoints', 1000)

# rrt.plan()
# rrt.plot()

# rrt = RRT(car, road, 'root', [50 22 0], 'npoints', 1000, 'simtime', 4)

# p = rrt.query([40 45 0], [50 22 0])

# about p

# rrt.plot(p)

# plot_vehicle(p, 'box', 'size', [20 30], 'fi ll', 'r', 'alpha', 0.1)
