#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from numpy import ndarray, eye, copy as npcopy, array
from spatialmath.base import r2q
from abc import ABC
from spatialgeometry.scene import node_init, node_update, scene_graph_children, scene_graph_tree
from spatialmath import SE3
from copy import deepcopy

# from roboticstoolbox.robot.ETS import ETS
from typing import Type, Union, List


class SceneNode:
    def __init__(
        self,
        T: ndarray = eye(4),
        scene_parent: Union["SceneNode", None] = None,
        scene_children: Union[List["SceneNode"], None] = None,
    ):
        # These three are static attributes which can never be changed
        # If these are directly accessed and re-written, segmentation faults
        # will follow very soon after
        # wT and sT cannot be accessed and set by users by base can be
        # modified through its setter

        # The world transform
        self.__wT = eye(4).copy(order="F")

        # The quaternion extracted from wT
        self.__wq = array([0.0, 0.0, 0.0, 1.0])

        # The local transform
        self.__T = eye(4).copy(order="F")
        self.__T[:] = T.copy(order="F")

        if scene_children is None:
            self._scene_children = []
        else:
            self._scene_children = scene_children

        self._scene_parent = scene_parent

        # Set up the c object
        self.__scene = self.__init_c()

        # Update childs parent
        for child in self.scene_children:
            child._update_scene_parent(self)

        # Update parents child
        if scene_parent is not None:
            scene_parent._update_scene_children(self)

        # Update scene tree
        self._propogate_scene_children()

    # --------------------------------------------------------------------- #

    def _custom_scene_node_init(
        self,
        T: ndarray = eye(4),
        scene_parent: Union["SceneNode", None] = None,
        scene_children: Union[List["SceneNode"], None] = None,
    ):
        # The world transform
        self.__wT = eye(4).copy(order="F")

        # The quaternion extracted from wT
        self.__wq = array([0.0, 0.0, 0.0, 1.0])

        # The local transform
        self.__T = eye(4).copy(order="F")
        self.__T[:] = T.copy(order="F")

        if scene_children is None:
            self._scene_children = []
        else:
            self._scene_children = scene_children

        self._scene_parent = scene_parent

        # Set up the c object
        self.__scene = self.__init_c()

        # Update childs parent
        for child in self.scene_children:
            child._update_scene_parent(self)

        # Update parents child
        if scene_parent is not None:
            scene_parent._update_scene_children(self)

        # Update scene tree
        self._propogate_scene_children()

    # --------------------------------------------------------------------- #

    def __init_c(self):
        """
        Super Private method which initialises a C object to hold Data

        """

        return node_init(
            len(self._scene_children),
            self.__T,
            self.__wT,
            self.__wq,
            self._scene_parent._scene if self._scene_parent is not None else None,
            [child._scene for child in self._scene_children],
        )

    def __update_c(self):
        """
        Super Private method which updates the C object which holds Data

        """

        node_update(
            self.__scene,
            len(self._scene_children),
            self._scene_parent._scene if self._scene_parent is not None else None,
            [child._scene for child in self._scene_children],
        )

    @property
    def _scene(self):
        return self.__scene

    # --------------------------------------------------------------------- #

    def __copy__(self):
        return deepcopy(self)

    def __deepcopy__(self, memo):
        result = SceneNode(
            T=self._T,
        )

        result._scene_children = self.scene_children.copy()
        result._scene_parent = self.scene_parent
        result.__update_c()
        memo[id(self)] = result
        return result

    def __str__(self) -> str:
        if self._scene_parent is not None:
            parent = f"{SE3(self._scene_parent._T, check=False).t}"
        else:
            parent = "None"

        return f"parent: {parent} \n self: {SE3(self._T).t} \n children: {[SE3(child._T).t for child in self._scene_children]}"

        # parent = self.scene_parent
        # return f"parent: {hex(id(parent)) if parent is not None else None} \n self: {hex(id(self))} \n children: {[hex(id(child)) for child in self.scene_children]}"

    # --------------------------------------------------------------------- #

    @property
    def scene_parent(self) -> Type["SceneNode"]:
        """
        Returns the parent node of this object

        """
        return self._scene_parent

    @scene_parent.setter
    def scene_parent(self, parent: "SceneNode"):
        """
        Sets a new parent node of this object, will automatically update
        the parents child

        """
        # Set our parent
        self._scene_parent = parent

        # Update our parents children
        parent._update_scene_children(self)

        # Update c
        self.__update_c()

    def _update_scene_parent(self, parent: "SceneNode"):
        """
        Sets a new parent node of this object, does NOT update
        the parents child

        """
        self._scene_parent = parent

        # Update c
        self.__update_c()

    # --------------------------------------------------------------------- #

    @property
    def scene_children(self) -> List["SceneNode"]:
        """
        Returns the child nodes of this object

        """
        return self._scene_children

    @scene_children.setter
    def scene_children(self, children: List["SceneNode"]):
        """
        Sets the child nodes of this object, does not update childs
        parent

        """
        # Set our children
        self._scene_children = children

        # Update our childrens parent
        for child in children:
            child._update_scene_parent(self)

        # Update c
        self.__update_c()

    def _update_scene_children(self, child: "SceneNode"):
        """
        Appends a new child to this object, does NOT update
        the childs parent

        """
        self.scene_children.append(child)

        # Update c
        self.__update_c()

    # --------------------------------------------------------------------- #

    @property
    def _wT(self) -> ndarray:
        """
        Returns the transform of this object in the world frame

        """
        return self.__wT

    @property
    def _wq(self) -> ndarray:
        """
        Returns the quaternion of this object in the world frame.

        """
        return self.__wq

    # --------------------------------------------------------------------- #

    @property
    def _T_reference(self) -> ndarray:
        """
        Returns the transform of this object with respect to the parent
        frame.

        """
        return self.__T

    @property
    def _T(self) -> ndarray:
        """
        Returns a copy of the transform of this object with respect to the parent
        frame.

        """
        return npcopy(self.__T)

    @_T.setter
    def _T(self, T: ndarray):
        self.__T[:] = T.copy(order="F")

        if self._scene_parent is not None:
            self.__wT[:] = self.parent.wT @ self._T
        else:
            self.__wT[:] = self._T

        self.__wq[:] = r2q(self.__wT[:3, :3], order="xyzs")

    # --------------------------------------------------------------------- #
    # Scene transform propogation methods
    #
    # The scene graph is a Forest -- A disjoint union of Rooted Trees
    # Each tree has a single root, no cycles, and each node has at most one
    # parent but unlimited children.
    # --------------------------------------------------------------------- #

    def _propogate_scene_children(self):
        """
        Propogates the world transform starting from this node going downwards
        through the tree (will not go through parents)
        """
        scene_graph_children(self.__scene)

    def _propogate_scene_tree(self):
        """
        Propogates the world transform starting from this root of the tree in
        which this node lives
        """
        scene_graph_tree(self.__scene)

    # --------------------------------------------------------------------- #

    def attach(self, object: "SceneNode"):
        new_childs = self.scene_children
        new_childs.append(object)
        self.scene_children = new_childs

    def attach_to(self, object: "SceneNode"):
        self.scene_parent = object
