Map classes
===========

Planner endpoints, returned paths and plot axes use world coordinates ``(x, y)``,
or ``(x, y, theta)`` with angles in radians. Distances and obstacle inflation
radii are in world units; parameters explicitly expressed in cells, such as
PRM's ``collision_step``, retain that meaning.

An occupancy grid's ``origin`` is the world position of cell ``(column=0, row=0)``,
and ``cellsize`` is the side length of a cell. Arrays are indexed as
``[row, column]``. Use ``world_to_grid`` (``w2g``) and ``grid_to_world`` (``g2w``)
to convert coordinates. A raw array has unit cells and origin ``(0, 0)``, as in
the book examples. DistanceTransformPlanner, DstarPlanner and Bug2 round
endpoints to the nearest cell and return cell centres in world coordinates;
continuous planners retain continuous world coordinates. Distance-transform
fields are in world units, while the standalone ``distancexform`` array
algorithm measures distances in cells. D* costs are per world unit, and its
sensor receives and returns positions in world coordinates.

A map draws its own background. An occupancy grid's ``plot`` method accepts an
optional ``field`` with the same shape as the grid, plus ``colorbar``,
``colorbar_style``, ``labels``, ``label_style``, ``cmap`` and an ``inflated``
obstacle mask. The existing ``map`` argument replaces the image without an
obstacle overlay. A polygon map draws its polygons in world coordinates.

Fields, obstacles and labels share the grid's coordinate conversion and the
supplied axes. Image edges extend half a cell beyond the outer cell centres;
labels sit at cell centres, and enabling them shows edge cells in full.
Plotting preserves the supplied field and style dictionaries.

Occupancy grid classes
----------------------

Binary Occupancy grid
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.BinaryOccupancyGrid
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__
   

Occupancy grid
^^^^^^^^^^^^^^

.. autoclass:: roboticstoolbox.mobile.OccupancyGrid
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

Polygon map
-----------

.. autoclass:: roboticstoolbox.mobile.PolygonMap
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__
