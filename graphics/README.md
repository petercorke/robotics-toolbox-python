This folder contains the graphics functionality of the toolbox.
Instructions on how to use the graphical section of the toolbox below.
(Pictures to come)

# TODO
 * Robot Joints
   * Rotational Joint
     * Rotate function
   * Prismatic Joint
     * Translate function
   * ~~Static Joint~~
   * **_POSTPONED_** Gripper Joint
 * Grid Updating
   * On rotation/move finish
   * ~~Don't redraw labels, move/update them~~
   * ~~Don't redraw the grid, move/update them~~
 * Error handling
   * ~~Throw custom error messages~~
   * Handle vpython error messages
 * STL
   * ~~Load Binary STL files~~
   * **_NOT POSSIBLE_** Option to save a mesh to STL?
     * Can't access object triangles/vertices.
     * Maybe just save vpython object details and recreate? won't allow compounds though. only basic entities.
 * 2D Graphics
   * Will likely not be done in vpython (overkill)

# Wish List
 * Updated Canvas Controls
   * WASD-QE controls (move/rotate)
   * Mouse rotation
 * Webpage Controls
   * Buttons that allow toggling display options
     * Labels, reference frames, robot, etc
 * Robot Interaction
   * Use the mouse/keyboard to manually rotate/move joints 

# How To
## Importing
To use the graphics, simply import it. For these examples, it is assumed the graphics are imported into the namespace 'gph'.
```python
import graphics as gph
```

## Common Functionality
VPython has its own data types that can be used. Firstly, the `radians()`, and `degrees()` functions convert between radians and degrees.
The `vector` class is also very crucial to the graphics. It can either represent a vector or a 3D point.

For convenience, some functions and variables are provided for easy use. `wrap_to_pi()` takes in an angle, and specification on degrees or radians. It returns the respective angle between -pi and pi.
Three vectors are also supplied for readability to ensure correct axes are used when referencing. `x_axis_vector`, `y_axis_vector`, `z_axis_vector` can be used to get vectors of an object, for example.
```python
# Wrap an angle (deg) to the range [-pi pi]. use "rad" instead of "deg" for radian angles.
gph.wrap_to_pi("deg", 450)
# Obtain the Z vector representation of the robot link
my_link.get_axis_vector(z_axis_vector)
```

## Setting Up The Scene
Any use of VPython objects requires a scene.

To create a scene to draw object to, a canvas must be created. Upon creation, a localhost http server will be opened. The function will return a GraphicsGrid object. 

Different attributes can be supplied to the function for some customisation. The display width, height, title, and caption can be manually input. Lastly, a boolean representing the grid visibility can be set.
```python
# Create a default canvas (1000*500, with grid displayed, no title or caption)
canvas_grid = gph.init_canvas()

# Alternatively create a grid with specified parameters
canvas_grid = gph.init_canvas(height=768, width=1024, title="Scene 1", caption="This scene shows...", grid=False)
``` 
The GraphicsGrid object has functions to update the visual, or to toggle visibility.
NB: `update_grid()` will be automated in future updates.
```python
# Update the grids to relocate/reorient to the camera focus point
# NB: Will be automated in future updates.
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


## Creating Robots
If you want to use the example puma560 robot, simply call the creation function that will return a `GraphicalRobot` object.
It will automatically be displayed in the canvas
```python
# Import the puma560 models and return a GraphicalRobot object
puma560 = gph.import_puma_560()
```
Otherwise, robots can be manually created using the `GraphicalRobot` class.
The joints for the robot can be manually or automatically created.

Firstly, create a `GraphicalRobot` object
```python
# Create an empty robot
my_robot = gph.GraphicalRobot()
```
Now we can add joints. The joints added to the robot act like a stack. First joints added will be last to be removed (if called to).

### Automatically
If you wish to automatically add joints, use `append_link()`. Add the joints from base to gripper.
This function takes in three arguments.

Firstly, the type of joint: rotational, prismatic, static, or gripper.
The input is the first letter of the type (case-insensitive). e.g. (rotational = "r" or "R").

Next is the initial pose (SE3 type) of the joint.

Lastly, the 'structure' of the robot. This variable must either be a `float` representing the joint length, or a `str`
representing a full file path to an STL file.

If a `float` length is given, a custom rectangular object will represent it in the scene. Otherwise if a `str` path is given,
the STL object will be loaded in and used in place of a rectangular joint.

```python
# Append a default base joint of length 2.
my_robot.append_link('r', SE3(), 2.0)

# Append an STL obj rotational joint.
my_robot.append_link('r', SE3(), './path/to/file.stl')
```

### Manually
Manually adding joints consists of creating a Joint object to add to the robot.
The types that can be created are identical to previously mentioned in the Automatic section.

`RotationalJoint`, `PrismaticJoint`, `StaticJoint`, `Gripper` are the class names of the different joints.

To create a joint, each class requires the same variables as the automatic version (minus the joint type string).

Although the creation process is "the same", manually creating a joint lets you more easily update any graphical issues associated with it.
For example, the STL you want to load may not be orientated/positioned correctly (How to fix is mentioned later)

```python
# Create two basic rotational links
link1 = gph.RotationalJoint(SE3(), 1.0)
link2 = gph.RotationalJoint(SE3(), 1.4)

# Add to the robot
my_robot.append_made_link(link1)
my_robot.append_made_link(link2)
``` 

### Deleting
To remove the end effector joint, use the `detach_link()` function. Acting like a stack, it will pop the latest joint created off the robot.

```python
# Remove the end effector joint
my_robot.detach_link()
```

## Importing an STL object

STL files may not be correctly positioned/oriented when loaded in.
Depending on where the object triangles are configured from the file, the origin of the object may not be where intended.

When loaded in, the origin is set (by default) to the center of the bounding box of the object.

Upon observation (through VPython or 3D editor software), you can find the coordinates of the origin in respect to the world.

A function `set_stl_joint_origin()` is supplied to change the origin.
This method is part of all joint types. It takes two 3D coordinates representing the world coordinates of where the desired origin currently is, and where the desired origin should be.

For example, if an STL object loads in and the origin is below (-z axis) where it should be, and the origin is at the bottom of the object, the following code will translate it up and set the origin.
```python
# Load the mesh in the link
link = gph.RotationalLink(SE3(), './path/to/file.stl')

# Obtain the graphical object to help with coordinates
# May not be necessary if you already know the 3D coordinates
stl_obj = link.get_graphic_object()

# Z origin is below where the current position is. (this e.g. is at bottom of object)
stl_obj_z_origin = stl_obj.pos.z - stl_obj.width / 2

# 3D pos of where the origin is
stl_obj_current_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, stl_obj_z_origin)

# 3D pos of where the origin should be set in the world
stl_obj_required_origin_location = vector(stl_obj.pos.x, stl_obj.pos.y, 0)

# Move the object to place origin where it should be, and apply the new origin to the object
link.set_stl_joint_origin(stl_obj_current_origin_location, stl_obj_required_origin_location)
```

## Using A GraphicalRobot
The robot class has two functions that handle the display. One function each to toggle the visibility for the joints and reference frames.
```python
# Turn off the robots reference frame displays
my_graphic_robot.set_reference_visibility(False)

# Toggle the robot visibility
my_graphic_robot.set_robot_visibility(not my_graphic_robot.is_shown)
``` 

To update the joint positions, use the `set_joint_poses()` function. It takes in a list of SE3 objects for each of the joints.
There must be 1 pose for each joint, in order from base to gripper (order of appending in creation)
```python
# Set all joint poses to a random configuration
# Assuming 3 joint robot
my_graphical_robot.set_joint_poses([
    SE3().Rand(),
    SE3().Rand(),
    SE3().Rand()
])
```

Alternatively, an `animate` function allows the robot to iterate through given poses to simulate movement.
Given an array of poses (per frame), and a frame rate, the robot will transition through each pose.
```python
my_graphical_robot.animate([
    [pose1, pose2, pose3],  # Frame 1
    [pose1, pose2, pose3],  # Frame 2
    [pose1, pose2, pose3],  # Frame 3
    [pose1, pose2, pose3]], # Frame 4
    4)  # 4 FPS
```

Lastly, a print function `print_joint_poses()` will print out the current poses of all joints.
```python
# Print joint poses
my_graphical_robot.print_joint_poses()
```
