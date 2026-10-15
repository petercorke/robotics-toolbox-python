# Mobile robot coordinates and map rendering

## Coordinates

All planner endpoints, returned paths and plot axes use world coordinates
`(x, y)`, or `(x, y, theta)` with angles in radians. Distances and inflation
radii are in world units. Parameters explicitly describing a sampling resolution
in cells, such as PRM's `collision_step`, retain that meaning.

An occupancy grid owns its coordinate conversion. Its `origin` is the world
position of cell `(column=0, row=0)`, and `cellsize` is the side length of a cell.
Array access remains `[row, column]`; use `world_to_grid` (`w2g`) and
`grid_to_world` (`g2w`) to cross that boundary. A raw array is a grid with unit
cells and origin `(0, 0)`, preserving the coordinates in the book examples.

DistanceTransformPlanner, DstarPlanner and Bug2 round endpoints to the nearest
cell using the grid's conversion and return cell centres in world coordinates.
Continuous planners, including PRMPlanner and RRTPlanner, retain continuous
world coordinates. DistanceTransformPlanner's distance field is in world units;
the standalone `distancexform` array algorithm still measures distances in cells.
D* costs are cost per world unit, and its sensor receives world coordinates and
returns `(x, y, cost)` updates in those same coordinates.

## Rendering

A map draws its own background. `BaseOccupancyGrid.plot` accepts an optional
`field`, with the grid's shape, plus `colorbar`, `colorbar_style`, `labels`,
`label_style`, `cmap` and an optional `inflated` obstacle mask. An existing `map`
argument still replaces the image, without an obstacle overlay. A `PolygonMap`
draws its polygons in world coordinates.

Image extents describe cell edges, half a cell beyond the outer cell centres.
Fields, obstacles and labels use the same transform and the supplied axes.
Labels are placed at cell centres; enabling them shows edge cells in full.
Plotting does not change the supplied field or style dictionaries.

`PlannerBase.plot_bg` supplies the original grid, an inflated mask if requested,
and the planner's distance field. Rendering details belong to the map. Existing
field colors, label styles and colorbar options are retained for the book's
unit-cell examples.

The `block` convention and animation arguments are separate concerns.
