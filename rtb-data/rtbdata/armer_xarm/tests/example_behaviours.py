#!/usr/bin/env python3
import rospy

# Behaviour Tree Packages
from ros_trees.trees import BehaviourTree
from ros_trees.leaves_common.console import Print
from ros_trees.leaves_ros import ActionLeaf, ServiceLeaf
from py_trees.composites import Sequence, Selector
from py_trees.decorators import FailureIsRunning, SuccessIsRunning, OneShot

# ROS messages
from geometry_msgs.msg import Pose

# Armer Specific Leaves/Branches
from armer_trees.motion import MoveToHomePose, MoveToNamedPose

from xarm_gripper.msg import MoveGoal


# Gripper action
class Gripper(ActionLeaf):
    """
    A ROS Action Leaf to open/close the xArm gripper
    """

    def __init__(
        self,
        name="Gripper Action",
        action_namespace="/xarm/gripper_move",
        open=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            name,
            action_namespace=action_namespace,
            load_fn=self.load_fn,
            *args,
            **kwargs,
        )
        self.open = open

    def load_fn(self):
        gripper = MoveGoal()
        if self.open:
            gripper.target_pulse = 800.0
        else:
            gripper.target_pulse = 5.0

        return gripper


if __name__ == "__main__":
    # Create ROS node
    rospy.init_node("xarm6_behaviour_test")

    # ----------- Initialise The Main Tree --------------------------------------------
    test_tree = BehaviourTree(
        "Xarm Test Agent",
        Sequence(
            name="Main Process",
            children=[
                # Move to underarm prep position
                MoveToNamedPose(load_value="xarm_real_test1", speed=0.1),
                # Gripper(
                #     open=False
                # ),
                # Move to underarm prep position
                MoveToNamedPose(load_value="xarm_real_test3", speed=0.1),
                # Gripper(
                #     open=True
                # ),
                MoveToHomePose(speed=0.1),
            ],
        ),
    )

    # Run the selected Tree
    test_tree.run(hz=30, push_to_start=False, setup_timeout=5, log_level="INFO")

    # test_tree.visualise()
