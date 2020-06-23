This folder contains the graphics functionality of the toolbox.
Instructions on how to use the graphical section of the toolbox below.
(Pictures to come)

# TODO
 * Robot Joints
   * Rotational Joint
   * Prismatic Joint
   * ~~Static Joint~~
   * Gripper Joint
 * Grid Updating
   * On rotation/move finish
   * ~~Don't redraw labels, move/update them~~
   * ~~Don't redraw the grid, move/update them~~
 * Error handling
   * ~~Throw custom error messages~~
   * Handle vpython error messages
 * STL
   * Load Binary STL files
   * Option to save a mesh to STL?
 * 2D Graphics
   * Will likely not be done in vpython (overkill)

# Future Additions
 * Updated Canvas Controls
   * WASD-QE controls (move/rotate)
   * Mouse rotation
 * Webpage Controls
   * Buttons that allow toggling display options
     * Labels, reference frames, robot, etc
 * Robot Interaction
   * Use the mouse/keyboard to manually rotate/move joints 

# How To
## Common Functionality
VPython has its own data types that have been used. Firstly, the `radians()`, and `degrees()` functions convert between radians and degrees.
The `vector` class is also very crucial to the graphics. It can either represent a vector or a 3D point.

For convenience, some functions and variables are provided for easy use. `wrap_to_pi()` takes in an angle, and specification on degrees or radians. It returns the respective angle between -pi and pi.
Three vectors are also supplied for readability to ensure correct axes are used when referencing. `x_axis_vector`, `y_axis_vector`, `z_axis_vector` can be used when supplying the rotation axis, for example.
```python
# Wrap an angle (deg) to the range [-pi pi]. use "rad" instead of "deg" for radian angles.
wrap_to_pi("deg", 450)
# Rotate the joint around its local x-axis by 30 degrees
rot_link.rotate_around_joint_axis(radians(30), x_axis_vector)
```

## Setting Up The Scene
Firstly, import the model_puma560 file to import all files/packages used within the graphics (imports are used within other files).
```python
from graphics.model_puma560 import *
```
Any use of VPython objects requires a scene.

To create a scene to draw object to, a canvas must be created. Upon creation, a localhost http server will be opened. The function will return a GraphicsGrid object. 

Different attributes can be supplied to the function for some customisation. The display width, height, title, and caption can be manually input. Lastly, a boolean representing the grid visibility can be set.
```python
# Create a default canvas (1000*500, with grid displayed, no title or caption)
canvas_grid = init_canvas()

# Alternatively create a grid with specified parameters
canvas_grid = init_canvas(height=768, width=1024, title="Scene 1", caption="This scene shows...", grid=False)
``` 
The GraphicsGrid object has functions to update the visual, or to toggle visibility.
```python
# Update the grids to relocate/reorient to the camera focus point
canvas_grid.update_grid()

# Turn off the visual display of the grid
canvas_grid.set_visibility(False)
```
Now that the scene is created, a robot must be created to be displayed.

At anytime you can clear the scene of all objects (The grid will remain if visible). Note: This will note delete the objects,
they still exist, and can be rendered visible afterwards. However, overwriting/deleting the variables will free the memory.
If an object is overwritten/deleted while still visible, the objects will remain in the scene.
```python
canvas_grid.clear_scene()
```

 

## Displaying Robot Joints
If you want to use the example puma560 robot, simply call the creation function that will return a GraphicalRobot object. It will automatically be displayed in the canvas
```python
# Import the puma560 models and return a GraphicalRobot object
puma560 = import_puma_560()
```

Creating your own robot is just as easy with the ability to use STL objects for your own custom robot, or using simple line segments.
Importing the STL objects is described below. Once you have the created 3D objects from the import, they can be used in the constructors.

Firstly, decide which type of joint you require: `RotationalJoint`, `PrismaticJoint`, `StaticJoint`, `Gripper`. 

All joint types can be supplied with an `x_axis` parameter. Defaulting to `x_axis_vector`, this variable simply states which direction in space the object's current x-axis is pointing in. This allows alignment of reference frames, for example that objects aren't used sideways.

Rotational joints have an independent attribute `rotation_axis` to assign which axis the joint rotates about, defaulting to `y_axis_vector`.

If using an STL object, the connection parameters are the 3D points in space (real coordinates of where the object is currently) that correspond to the positions where the neighbour joints connect to it.
For example, if the object is currently loaded in with the point where it would connect to a previous segment at (0, 0, 0), and the 'tooltip' or point it would connect to the next joint at (1, 0, 0), the creation would look like
```python
# Current connection points
connect_from = vector(0, 0, 0)
connect_to = vector(1, 0, 0)

# Create a Rotational joint that is currently facing in the +x direction, that rotates about it's y-axis
new_link = RotationalJoint(connect_from,
                           connect_to,
                           x_axis=x_axis_vector,
                           rotation_axis=y_axis_vector,
                           graphic_obj=your_stl_obj)
``` 
Otherwise if no prior 3D object is given, a line segment will be created to render as the joint. The `x_axis` attribute is not used in this situation.
```python
# The current joint will go from (1, 1, 0) to (1, 1, 3)
connect_from = vector(1, 1, 0)
connect_to = vector(1, 1, 3)

# Create a basic rotation joint
new_link = RotationalJoint(connect_from, connect_to)
``` 

The other joint types are created in the same way
```python
# Connection points
connect_from = vector(3, 0, 0)
connect_to = vector(0, 0, 0)

# Create a prismatic joint with our without an STL object
new_link_graphic = PrismaticJoint(connect_from,
                               connect_to,
                               x_axis=x_axis_vector,
                               graphic_obj=your_stl_obj)
new_link_basic = PrismaticJoint(connect_from, connect_to)

# Create a static joint with our without an STL object
new_link_graphic = StaticJoint(connect_from,
                               connect_to,
                               x_axis=x_axis_vector,
                               graphic_obj=your_stl_obj)
new_link_basic = StaticJoint(connect_from, connect_to)

# Create a gripper joint with our without an STL object
new_link_graphic = Gripper(connect_from,
                           connect_to,
                           x_axis=x_axis_vector,
                           graphic_obj=your_stl_obj)
new_link_basic = Gripper(connect_from, connect_to)
```
## Importing an STL
Importing an STL can either be straight-forward or a bit tedious. Firstly, import the STL file into VPython using `import_object_from_stl()`.
This will create a compound object from the triangles, and display it in the canvas.
```python
# Create a compound object from an STL file 'my_object'
# The search path is relative to ./graphics/models/
# Only the filename is required (no extension)
my_mesh = import_object_from_stl('my_object')
```
Then depending on where the object triangles are configured from the file, it may need to be translated or rotated.

The Joint classes assume that the origin of the joint is the rotation point. However, creating a compound object puts the origin at the centre of the 3D bounding box.
Since these positions may not align, translation and/or rotation may be required. 

If the loaded object was not oriented correctly upon loading, it can be manually rotated (preferably before setting the origin described below).
Manual inspection of the objects orientation will guide to how to rotate the object in space. They can be easily rotated through VPython's object rotate function
```python
# Load the mesh
my_stl_obj = import_object_from_stl('my_object')

# Rotate the object about the +Y axis 90 degrees
my_stl_obj.rotate(angle=radians(90), axis=y_axis_vector, origin=vector(0, 0, 0))
```
A function `set_stl_origin()` is also supplied to change the origin.
This function takes in a graphical object, and two 3D points representing the world coordinates of where the desired origin currently is, and where the desired origin should be.

For example, if an STL object loads in and the origin is below (-z) where it should be, and the origin is at the bottom of the object, the following code will translate it up and set the origin.
```python
# Load the mesh
my_stl_obj = import_object_from_stl('my_object')

# Find the coordinates of where the desired origin is 
# It's at the bottom of the object, that is entirely below the z=0 plane

# Z coordinate is located from the middle of the object, with an extra distance of half the object away.
my_stl_obj_z_pos = my_stl_obj.pos.z - my_stl_obj.width/2
# X and Y coordinates already in place 
current_origin_location = vector(my_stl_obj.pos.x, my_stl_obj.pos.y, my_stl_obj_z_pos)
# Origin should be at current X, Y, with Z at 0
required_origin_location = vector(my_stl_obj.pos.x, my_stl_obj.pos.y, 0)

# Update the STL object
my_stl_obj = set_stl_origin(my_stl_obj, current_origin_location, required_origin_location)
```
The STL objects can now be used in the Joint classes without hassle.

## Creating a GraphicalRobot
Now that you have created all of the robot links, a `GraphicalRobot` can be created. Simply inputting a list of the joints to the constructor is all that is required.

The order of joints is important! It is assumed that index 0 is the base, and incrementally goes through the robot from base to tooltip.
```python
# Create a default 3-link rotational robot along the +X axis.
my_graphic_robot = GraphicalRobot([
    RotationalJoint(vector(0, 0, 0), vector(1, 0, 0)),
    RotationalJoint(vector(1, 0, 0), vector(2, 0, 0)),
    RotationalJoint(vector(2, 0, 0), vector(3, 0, 0))
])
```
Or if 3D object meshes are used for links
```python
# Base Joint: Rotate +z axis, facing x-axis (def)
base_joint = RotationalJoint(vector(0, 0, 0), vector(1, 0, 0), rotation_axis=z_axis_vector, graphic_obj=my_stl_object1),
# Middle Joint: Rotate +y axis (def), facing x-axis (def)
mid_joint = RotationalJoint(vector(1, 0, 0), vector(2, 0, 0), graphic_obj=my_stl_object2),
# End Joint: Rotate +y axis (def), facing x-axis (def)
end_joint = RotationalJoint(vector(2, 0, 0), vector(3, 0, 0), graphic_obj=my_stl_object3),

# Create a 3D mesh graphical robot
my_graphic_robot = GraphicalRobot([
    base_joint,
    mid_joint,
    end_joint
])
```

## Using A GraphicalRobot
The robot class has two functions that handle the display. One function each to toggle the visibility for the joints and reference frames.
```python
# Turn off the robots reference frame displays
my_graphic_robot.set_reference_visibility(False)

# Toggle the robot visibility
my_graphic_robot.set_robot_visibility(not my_graphic_robot.is_shown)
``` 
Moving the robot around in the 3D space is possible through `move_base()`. Given a 3D coordinate, the origin of the base will be relocated to this position.
```python
# Move the base of the robot to (2, 3, 0)
my_graphic_robot.move_base(vector(2, 3, 0))
```
Setting joint angles can be done in two ways. Firstly, one joint can be rotated individually.
The function takes the joint index (from the list of creation) and an angle (radians) to set the joint angle to. The angle given is it's local rotation angle.

If a joint that is not a rotational joint is given, an error will be displayed.
```python
joint_index = 1
new_angle = radians(30)

# Rotate the 1st joint of the robot (base = 0) to an angle of 30 degrees
my_graphic_robot.set_joint_angle(joint_index, new_angle)
```
Otherwise, all joint angles can be modified together.
If the given list of angles doesn't match the number of joints, an error will be displayed.
Further, while iterating through the joints, a message will appear saying a non-revolute was found. It will skip this joint and it's associated given angle.
```python
# Assuming a 3-joint all-rotational robot
my_graphical_robot.set_all_joint_angles([
    radians(-45),
    radians(45),
    radians(15)
])
```

Lastly, a print function `print_joint_angles()` will print out the current local joint angles, if a revolute.
If the output angles are to be displayed in degrees, True should be input as a parameter.
```python
# Print joint angles in radians
my_graphical_robot.print_joint_angles()
# Print joint angles in degrees
my_graphical_robot.print_joint_angles(True)
```
