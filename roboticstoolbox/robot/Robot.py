#!/usr/bin/env python

"""
@author: Jesse Haviland
@author: Peter Corke
"""

# import sys
from os.path import splitext
from copy import deepcopy
from warnings import warn
from pathlib import PurePosixPath, Path
from typing import (
    List,
    TypeVar,
    Union,
    Dict,
    Tuple,
    overload,
)
from typing_extensions import Literal as L


import numpy as np

import spatialmath.base as smb
from spatialmath.base.argcheck import (
    getvector,
    getmatrix,
    verifymatrix,
)

from spatialgeometry import Shape, CollisionShape, Cylinder

from spatialmath import (
    SE3,
    SE2,
    SpatialAcceleration,
    SpatialVelocity,
    SpatialInertia,
    SpatialForce,
)

import roboticstoolbox as rtb
from roboticstoolbox.robot.BaseRobot import BaseRobot
from roboticstoolbox.robot.RobotKinematics import RobotKinematicsMixin
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.robot.Link import BaseLink, Link, Link2
from roboticstoolbox.robot.ETS import ETS, ETS2
from roboticstoolbox.tools import xacro
from roboticstoolbox.tools import URDF
from roboticstoolbox.tools.types import ArrayLike, NDArray
from roboticstoolbox.tools.data import rtb_path_to_datafile

# A generic type variable representing any subclass of BaseLink
LinkType = TypeVar("LinkType", bound=BaseLink)


# ==================================================================================== #
# ================= Robot Class ====================================================== #
# ==================================================================================== #


class Robot(BaseRobot[Link], RobotKinematicsMixin):
    _color = True

    def __init__(
        self,
        arg: Union[List[Link], ETS, "Robot"],
        gripper_links: Union[Link, List[Link], None] = None,
        name: str = "",
        manufacturer: str = "",
        comment: str = "",
        base: Union[NDArray, SE3, None] = None,
        tool: Union[NDArray, SE3, None] = None,
        gravity: ArrayLike = [0, 0, -9.81],
        keywords: Union[List[str], Tuple[str]] = [],
        symbolic: bool = False,
        configs: Union[Dict[str, NDArray], None] = None,
        check_jindex: bool = True,
        urdf_string: Union[str, None] = None,
        urdf_filepath: Union[Path, PurePosixPath, None] = None,
    ):
        # Process links
        if isinstance(arg, Robot):
            # We're passed a Robot, clone it
            # We need to preserve the parent link as we copy

            # Copy each link within the robot
            links = [deepcopy(link) for link in arg.links]
            gripper_links = []

            for gripper in arg.grippers:
                glinks = []
                for link in gripper.links:
                    glinks.append(deepcopy(link))

                gripper_links.append(glinks[0])
                links = links + glinks

            # Sever parent connection, but save the string
            # The constructor will piece this together for us
            for link in links:
                link._children = []
                if link.parent is not None:
                    link._parent_name = link.parent.name
                    link._parent = None

            super().__init__(links, gripper_links=gripper_links)

            for i, gripper in enumerate(self.grippers):
                gripper.tool = arg.grippers[i].tool.copy()

            self._urdf_string = arg.urdf_string
            self._urdf_filepath = arg.urdf_filepath
        else:
            if isinstance(arg, ETS):
                # We're passed an ETS string
                links = []
                # chop it up into segments, a link frame after every joint
                parent = None
                for j, ets_j in enumerate(arg.split()):
                    elink = Link(ETS(ets_j), parent=parent, name=f"link{j:d}")
                    if (
                        elink.qlim is None
                        and elink.v is not None
                        and elink.v.qlim is not None
                    ):
                        elink.qlim = elink.v.qlim  # pragma nocover
                    parent = elink
                    links.append(elink)

            elif smb.islistof(arg, Link):
                links = arg

            else:
                raise TypeError("arg was invalid, must be List[Link], ETS, or Robot")

            # Initialise Base Robot object
            super().__init__(
                links=links,
                gripper_links=gripper_links,
                name=name,
                manufacturer=manufacturer,
                comment=comment,
                base=base,
                tool=tool,
                gravity=gravity,
                keywords=keywords,
                symbolic=symbolic,
                configs=configs,
                check_jindex=check_jindex,
            )

    # --------------------------------------------------------------------- #
    # --------- Swift Methods --------------------------------------------- #
    # --------------------------------------------------------------------- #

    def _to_dict(self, robot_alpha=1.0, collision_alpha=0.0):
        ob = []

        for link in self.links:
            if robot_alpha > 0:
                for gi in link.geometry:
                    gi.set_alpha(robot_alpha)
                    ob.append(gi.to_dict())
            if collision_alpha > 0:
                for gi in link.collision:
                    gi.set_alpha(collision_alpha)
                    ob.append(gi.to_dict())

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:
                if robot_alpha > 0:
                    for gi in link.geometry:
                        gi.set_alpha(robot_alpha)
                        ob.append(gi.to_dict())
                if collision_alpha > 0:
                    for gi in link.collision:
                        gi.set_alpha(collision_alpha)
                        ob.append(gi.to_dict())

        # for o in ob:
        #     print(o)

        return ob

    def _fk_dict(self, robot_alpha=1.0, collision_alpha=0.0):
        ob = []

        # Do the robot
        for link in self.links:
            if robot_alpha > 0:
                for gi in link.geometry:
                    ob.append(gi.fk_dict())
            if collision_alpha > 0:
                for gi in link.collision:
                    ob.append(gi.fk_dict())

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:
                if robot_alpha > 0:
                    for gi in link.geometry:
                        ob.append(gi.fk_dict())
                if collision_alpha > 0:
                    for gi in link.collision:
                        ob.append(gi.fk_dict())

        return ob

    # --------------------------------------------------------------------- #
    # --------- URDF Methods ---------------------------------------------- #
    # --------------------------------------------------------------------- #

    @staticmethod
    def URDF_read(
        file_path, tld=None, xacro_tld=None
    ) -> Tuple[List[Link], str, str, Union[Path, PurePosixPath]]:
        """
        Read a URDF file as Links

        File should be specified relative to ``RTBDATA/URDF/xacro``

        Parameters
        ----------
        file_path
            File path relative to the xacro folder
        tld
            A custom top-level directory which holds the xacro data,
            defaults to None
        xacro_tld
            A custom top-level within the xacro data,
            defaults to None

        Returns
        -------
        links
            a list of links
        name
            the name of the robot
        urdf
            a string representing the URDF
        file_path
            a path to the original file

        Notes
        -----
        If ``tld`` is not supplied, filepath pointing to xacro data should
        be directly under ``RTBDATA/URDF/xacro`` OR under ``./xacro`` relative
        to the model file calling this method. If ``tld`` is supplied, then
        ```file_path``` needs to be relative to ``tld``

        """

        # Get the path to the class that defines the robot
        if tld is None:
            base_path = rtb_path_to_datafile("xacro")
        else:
            base_path = PurePosixPath(tld)

        # Add on relative path to get to the URDF or xacro file
        # base_path = PurePath(classpath).parent.parent / 'URDF' / 'xacro'
        file_path = base_path / PurePosixPath(file_path)
        _, ext = splitext(file_path)

        if ext == ".xacro":
            # it's a xacro file, preprocess it
            if xacro_tld is not None:
                xacro_tld = base_path / PurePosixPath(xacro_tld)
            urdf_string = xacro.main(file_path, xacro_tld)
            try:
                urdf = URDF.loadstr(urdf_string, file_path, base_path)
            except BaseException as e:  # pragma nocover
                print("error parsing URDF file", file_path)
                raise e
        else:  # pragma nocover
            urdf_string = open(file_path).read()
            urdf = URDF.loadstr(urdf_string, file_path, base_path)

        if not isinstance(urdf_string, str):  # pragma nocover
            raise ValueError("Parsing failed, did not get valid URDF string back")

        return urdf.elinks, urdf.name, urdf_string, file_path

    @classmethod
    def URDF(cls, file_path: str, gripper: Union[int, str, None] = None):
        """
        Construct a Robot object from URDF file

        Parameters
        ----------
        file_path
            the path to the URDF
        gripper
            index or name of the gripper link(s)

        Returns
        -------
        If ``gripper`` is specified, links from that link outward are removed
        from the rigid-body tree and folded into a ``Gripper`` object.

        """

        links, name, urdf_string, urdf_filepath = Robot.URDF_read(file_path)

        gripperLink: Union[Link, None] = None

        if gripper is not None:
            if isinstance(gripper, int):
                gripperLink = links[gripper]
            elif isinstance(gripper, str):
                for link in links:
                    if link.name == gripper:
                        gripperLink = link
                        break
                else:  # pragma nocover
                    raise ValueError(f"no link named {gripper}")
            else:  # pragma nocover
                raise TypeError("bad argument passed as gripper")

        # links, name, urdf_string, urdf_filepath = Robot.URDF_read(file_path)

        return cls(
            links,
            name=name,
            gripper_links=gripperLink,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

    #     # --------------------------------------------------------------------- #
    #     # --------- Utility Methods ------------------------------------------- #
    #     # --------------------------------------------------------------------- #

    #     def showgraph(self, display_graph: bool = True, **kwargs) -> Union[None, str]:
    #         """
    #         Display a link transform graph in browser

    #         ``robot.showgraph()`` displays a graph of the robot's link frames
    #         and the ETS between them.  It uses GraphViz dot.

    #         The nodes are:
    #             - Base is shown as a grey square. This is the world frame origin,
    #               but can be changed using the ``base`` attribute of the robot.
    #             - Link frames are indicated by circles
    #             - ETS transforms are indicated by rounded boxes

    #         The edges are:
    #             - an arrow if `jtype` is False or the joint is fixed
    #             - an arrow with a round head if `jtype` is True and the joint is
    #               revolute
    #             - an arrow with a box head if `jtype` is True and the joint is
    #               prismatic

    #         Edge labels or nodes in blue have a fixed transformation to the
    #         preceding link.

    #         Parameters
    #         ----------
    #         display_graph
    #             Open the graph in a browser if True. Otherwise will return the
    #             file path
    #         etsbox
    #             Put the link ETS in a box, otherwise an edge label
    #         jtype
    #             Arrowhead to node indicates revolute or prismatic type
    #         static
    #             Show static joints in blue and bold

    #         Examples
    #         --------
    #         >>> import roboticstoolbox as rtb
    #         >>> panda = rtb.models.URDF.Panda()
    #         >>> panda.showgraph()

    #         .. image:: ../figs/panda-graph.svg
    #             :width: 600

    #         See Also
    #         --------
    #         :func:`dotfile`

    #         """

    #         # Lazy import
    #         import tempfile
    #         import subprocess
    #         import webbrowser

    #         # create the temporary dotfile
    #         dotfile = tempfile.TemporaryFile(mode="w")
    #         self.dotfile(dotfile, **kwargs)

    #         # rewind the dot file, create PDF file in the filesystem, run dot
    #         dotfile.seek(0)
    #         pdffile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    #         subprocess.run("dot -Tpdf", shell=True, stdin=dotfile, stdout=pdffile)

    #         # open the PDF file in browser (hopefully portable), then cleanup
    #         if display_graph:  # pragma nocover
    #             webbrowser.open(f"file://{pdffile.name}")
    #         else:
    #             return pdffile.name

    #     def dotfile(
    #         self,
    #         filename: Union[str, IO[str]],
    #         etsbox: bool = False,
    #         ets: L["full", "brief"] = "full",
    #         jtype: bool = False,
    #         static: bool = True,
    #     ):
    #         """
    #         Write a link transform graph as a GraphViz dot file

    #         The file can be processed using dot:
    #             % dot -Tpng -o out.png dotfile.dot

    #         The nodes are:
    #             - Base is shown as a grey square.  This is the world frame origin,
    #               but can be changed using the ``base`` attribute of the robot.
    #             - Link frames are indicated by circles
    #             - ETS transforms are indicated by rounded boxes

    #         The edges are:
    #             - an arrow if `jtype` is False or the joint is fixed
    #             - an arrow with a round head if `jtype` is True and the joint is
    #               revolute
    #             - an arrow with a box head if `jtype` is True and the joint is
    #               prismatic

    #         Edge labels or nodes in blue have a fixed transformation to the
    #         preceding link.

    #         Note
    #         ----
    #         If ``filename`` is a file object then the file will *not*
    #             be closed after the GraphViz model is written.

    #         Parameters
    #         ----------
    #         file
    #             Name of file to write to
    #         etsbox
    #             Put the link ETS in a box, otherwise an edge label
    #         ets
    #             Display the full ets with "full" or a brief version with "brief"
    #         jtype
    #             Arrowhead to node indicates revolute or prismatic type
    #         static
    #             Show static joints in blue and bold

    #         See Also
    #         --------
    #         :func:`showgraph`

    #         """

    #         if isinstance(filename, str):
    #             file = open(filename, "w")
    #         else:
    #             file = filename

    #         header = r"""digraph G {
    # graph [rankdir=LR];
    # """

    #         def draw_edge(link, etsbox, jtype, static):
    #             # draw the edge
    #             if jtype:
    #                 if link.isprismatic:
    #                     edge_options = 'arrowhead="box", arrowtail="inv", dir="both"'
    #                 elif link.isrevolute:
    #                     edge_options = 'arrowhead="dot", arrowtail="inv", dir="both"'
    #                 else:
    #                     edge_options = 'arrowhead="normal"'
    #             else:
    #                 edge_options = 'arrowhead="normal"'

    #             if link.parent is None:
    #                 parent = "BASE"
    #             else:
    #                 parent = link.parent.name

    #             if etsbox:
    #                 # put the ets fragment in a box
    #                 if not link.isjoint and static:
    #                     node_options = ', fontcolor="blue"'
    #                 else:
    #                     node_options = ""

    #                 try:
    #                     file.write(
    #                         '  {}_ets [shape=box, style=rounded, label="{}"{}];\n'.format(
    #                             link.name,
    #                             link.ets.__str__(q=f"q{link.jindex}"),
    #                             node_options,
    #                         )
    #                     )
    #                 except UnicodeEncodeError:  # pragma nocover
    #                     file.write(
    #                         '  {}_ets [shape=box, style=rounded, label="{}"{}];\n'.format(
    #                             link.name,
    #                             link.ets.__str__(q=f"q{link.jindex}")
    #                             .encode("ascii", "ignore")
    #                             .decode("ascii"),
    #                             node_options,
    #                         )
    #                     )

    #                 file.write("  {} -> {}_ets;\n".format(parent, link.name))
    #                 file.write(
    #                     "  {}_ets -> {} [{}];\n".format(link.name, link.name, edge_options)
    #                 )
    #             else:
    #                 # put the ets fragment as an edge label
    #                 if not link.isjoint and static:
    #                     edge_options += 'fontcolor="blue"'
    #                 if ets == "full":
    #                     estr = link.ets.__str__(q=f"q{link.jindex}")
    #                 elif ets == "brief":
    #                     if link.jindex is None:
    #                         estr = ""
    #                     else:
    #                         estr = f"...q{link.jindex}"
    #                 else:
    #                     return
    #                 try:
    #                     file.write(
    #                         '  {} -> {} [label="{}", {}];\n'.format(
    #                             parent,
    #                             link.name,
    #                             estr,
    #                             edge_options,
    #                         )
    #                     )
    #                 except UnicodeEncodeError:  # pragma nocover
    #                     file.write(
    #                         '  {} -> {} [label="{}", {}];\n'.format(
    #                             parent,
    #                             link.name,
    #                             estr.encode("ascii", "ignore").decode("ascii"),
    #                             edge_options,
    #                         )
    #                     )

    #         file.write(header)

    #         # add the base link
    #         file.write("  BASE [shape=square, style=filled, fillcolor=gray]\n")

    #         # add the links
    #         for link in self:
    #             # draw the link frame node (circle) or ee node (doublecircle)
    #             if link in self.ee_links:
    #                 # end-effector
    #                 node_options = 'shape="doublecircle", color="blue", fontcolor="blue"'
    #             else:
    #                 node_options = 'shape="circle"'

    #             file.write("  {} [{}];\n".format(link.name, node_options))

    #             draw_edge(link, etsbox, jtype, static)

    #         for gripper in self.grippers:
    #             for link in gripper.links:
    #                 file.write("  {} [shape=cds];\n".format(link.name))
    #                 draw_edge(link, etsbox, jtype, static)

    #         file.write("}\n")

    #         if isinstance(filename, str):
    #             file.close()  # noqa

    # --------------------------------------------------------------------- #
    # --------- Kinematic Methods ----------------------------------------- #
    # --------------------------------------------------------------------- #

    @property
    def reach(self) -> float:
        r"""
        Reach of the robot

        A conservative estimate of the reach of the robot. It is computed as
        the sum of the translational ETs that define the link transform.

        Note
        ----
        Computed on the first access. If kinematic parameters
        subsequently change this will not be reflected.

        Returns
        -------
        reach
            Maximum reach of the robot

        Notes
        -----
        - Probably an overestimate of reach
        - Used by numerical inverse kinematics to scale translational
          error.
        - For a prismatic joint, uses ``qlim`` if it is set

        """

        # TODO
        # This should be a start, end method and compute the reach based on the
        # given ets. Then use an lru_cache to speed up return

        if self._reach is None:
            d_all = []
            for link in self.ee_links:
                d = 0
                while True:
                    for et in link.ets:
                        if et.istranslation:
                            if et.isjoint:
                                # the length of a prismatic joint depends on the
                                # joint limits.  They might be set in the ET
                                # or in the Link depending on how the robot
                                # was constructed
                                if link.qlim is not None:
                                    d += max(link.qlim)
                                elif et.qlim is not None:  # pragma nocover
                                    d += max(et.qlim)
                            else:
                                d += abs(et.eta)
                    link = link.parent
                    if link is None or isinstance(link, str):
                        d_all.append(d)
                        break

            self._reach = max(d_all)
        return self._reach

    def fkine_all(self, q: ArrayLike) -> SE3:
        """
        Compute the pose of every link frame

        ``T = robot.fkine_all(q)`` is  an SE3 instance with ``robot.nlinks +
        1`` values:

        - ``T[0]`` is the base transform
        - ``T[i]`` is the pose of link whose ``number`` is ``i``

        Parameters
        ----------
        q
            The joint configuration

        Returns
        -------
        fkine_all
            Pose of all links

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """  # noqa

        q = getvector(q)

        Tbase = SE3(self.base)  # add base, also sets the type

        linkframes = Tbase.__class__.Alloc(self.nlinks + 1)
        linkframes[0] = Tbase

        def recurse(Tall, Tparent, q, link):
            # if joint??
            T = Tparent
            while True:
                T *= SE3(link.A(q[link.jindex]))

                Tall[link.number] = T

                if link.nchildren == 0:
                    # no children
                    return
                elif link.nchildren == 1:
                    # one child
                    if link in self.ee_links:  # pragma nocover
                        # this link is an end-effector, go no further
                        return
                    link = link.children[0]
                    continue
                else:
                    # multiple children
                    for child in link.children:
                        recurse(Tall, T, q, child)
                    return

        recurse(linkframes, Tbase, q, self.links[0])

        return linkframes

    @overload
    def manipulability(
        self,
        q: ArrayLike = ...,
        J: None = None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        method: L["yoshikawa", "asada", "minsingular", "invcondition"] = "yoshikawa",
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
        **kwargs,
    ) -> Union[float, NDArray]:  # pragma nocover
        ...

    @overload
    def manipulability(
        self,
        q: None = None,
        J: NDArray = ...,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        method: L["yoshikawa", "asada", "minsingular", "invcondition"] = "yoshikawa",
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
        **kwargs,
    ) -> Union[float, NDArray]:  # pragma nocover
        ...

    def manipulability(
        self,
        q=None,
        J=None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        method: L["yoshikawa", "asada", "minsingular", "invcondition"] = "yoshikawa",
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
        **kwargs,
    ):
        """
        Manipulability measure

        ``manipulability(q)`` is the scalar manipulability index
        for the robot at the joint configuration ``q``.  It indicates
        dexterity, that is, how well conditioned the robot is for motion
        with respect to the 6 degrees of Cartesian motion.  The values is
        zero if the robot is at a singularity.

        Parameters
        ----------
        q
            Joint coordinates, one of J or q required
        J
            Jacobian in base frame if already computed, one of J or
            q required
        method
            method to use, "yoshikawa" (default), "invcondition",
            "minsingular"  or "asada"
        axes
            Task space axes to consider: "all" [default],
            "trans", or "rot"

        Returns
        -------
        manipulability

        Synopsis
        --------

        Various measures are supported:

        | Measure           |       Description                               |
        |-------------------|-------------------------------------------------|
        | ``"yoshikawa"``   | Volume of the velocity ellipsoid, *distance*    |
        |                   | from singularity [Yoshikawa85]_                 |
        | ``"invcondition"``| Inverse condition number of Jacobian, isotropy  |
        |                   | of the velocity ellipsoid [Klein87]_            |
        | ``"minsingular"`` | Minimum singular value of the Jacobian,         |
        |                   | *distance*  from singularity [Klein87]_         |
        | ``"asada"``       | Isotropy of the task-space acceleration         |
        |                   | ellipsoid which is a function of the Cartesian  |
        |                   | inertia matrix which depends on the inertial    |
        |                   | parameters [Asada83]_                           |

        **Trajectory operation**:

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        Notes
        -----
        - Invokes the ``jacob0`` method of the robot if ``J`` is not passed
        - The "all" option includes rotational and translational
            dexterity, but this involves adding different units. It can be
            more useful to look at the translational and rotational
            manipulability separately.
        - Examples in the RVC book (1st edition) can be replicated by
            using the "all" option
        - Asada's measure requires inertial a robot model with inertial
            parameters.

        References
        ----------
        .. [Yoshikawa85] Manipulability of Robotic Mechanisms. Yoshikawa T.,
                The International Journal of Robotics Research.
                1985;4(2):3-9. doi:10.1177/027836498500400201
        .. [Asada83] A geometrical representation of manipulator dynamics and
                its application to arm design, H. Asada,
                Journal of Dynamic Systems, Measurement, and Control,
                vol. 105, p. 131, 1983.
        .. [Klein87] Dexterity Measures for the Design and Control of
                Kinematically Redundant Manipulators. Klein CA, Blaho BE.
                The International Journal of Robotics Research.
                1987;6(2):72-83. doi:10.1177/027836498700600206
        - Robotics, Vision & Control, Chap 8, P. Corke, Springer 2011.


        .. versionchanged:: 1.0.3
            Removed 'both' option for axes, added a custom list option.

        """

        ets = self.ets(end, start)

        axes_list: List[bool] = []

        if isinstance(axes, list):
            axes_list = axes
        elif axes == "all":
            axes_list = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes_list = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes_list = [False, False, False, True, True, True]
        elif axes == "both":
            return (
                self.manipulability(
                    q=q, J=J, end=end, start=start, method=method, axes="trans"
                ),
                self.manipulability(
                    q=q, J=J, end=end, start=start, method=method, axes="rot"
                ),
            )
        else:
            raise ValueError("axes must be all, trans, rot or both")

        def yoshikawa(robot, J, q, axes_list):
            J = J[axes_list, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(np.linalg.det(J))
            else:
                m2 = np.linalg.det(J @ J.T)
                return np.sqrt(abs(m2))

        def condition(robot, J, q, axes_list):
            J = J[axes_list, :]

            # return 1/cond(J)
            return 1 / np.linalg.cond(J)

        def minsingular(robot, J, q, axes_list):
            J = J[axes_list, :]
            s = np.linalg.svd(J, compute_uv=False)

            # return last/smallest singular value of J
            return s[-1]

        def asada(robot, J, q, axes_list):
            # dof = np.sum(axes_list)
            if np.linalg.matrix_rank(J) < 6:
                return 0
            Ji = np.linalg.pinv(J)
            Mx = Ji.T @ robot.inertia(q) @ Ji
            d = np.where(axes_list)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = np.linalg.eig(Mx)
            return np.min(e) / np.max(e)

        # choose the handler function
        if method.lower().startswith("yoshi"):
            mfunc = yoshikawa
        elif method.lower().startswith("invc"):
            mfunc = condition
        elif method.lower().startswith("mins"):
            mfunc = minsingular
        elif method.lower().startswith("asa"):
            mfunc = asada
        else:
            raise ValueError("Invalid method chosen")

        # Calculate manipulability based on supplied Jacobian
        if J is not None:
            w = [mfunc(self, J, q, axes_list)]

        # Otherwise use the q vector/matrix
        else:
            q = np.array(getmatrix(q, (None, self.n)))
            w = np.zeros(q.shape[0])

            for k, qk in enumerate(q):
                Jk = ets.jacob0(qk)
                w[k] = mfunc(self, Jk, qk, axes_list)

        if len(w) == 1:
            return w[0]
        else:
            return w

    def jtraj(
        self,
        T1: Union[NDArray, SE3],
        T2: Union[NDArray, SE3],
        t: Union[NDArray, int],
        **kwargs,
    ):
        """
        Joint-space trajectory between SE(3) poses

        The initial and final poses are mapped to joint space using inverse
        kinematics:

        - if the object has an analytic solution ``ikine_a`` that will be used,
        - otherwise the general numerical algorithm ``ikine_lm`` will be used.

        ``traj = obot.jtraj(T1, T2, t)`` is a trajectory object whose
        attribute ``traj.q`` is a row-wise joint-space trajectory.

        Parameters
        ----------
        T1
            initial end-effector pose
        T2
            final end-effector pose
        t
            time vector or number of steps
        kwargs
            arguments passed to the IK solver

        Returns
        -------
        trajectory

        """  # noqa

        if hasattr(self, "ikine_a"):
            ik = self.ikine_a  # type: ignore
        else:
            ik = self.ikine_LM

        q1 = ik(T1, **kwargs)
        q2 = ik(T2, **kwargs)

        return rtb.jtraj(q1.q, q2.q, t)

    @overload
    def jacob0_dot(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        J0: None = None,
        representation: Union[L["rpy/xyz", "rpy/zyx", "eul", "exp"], None] = None,
    ) -> NDArray:  # pragma no cover
        ...

    @overload
    def jacob0_dot(
        self,
        q: None,
        qd: ArrayLike,
        J0: NDArray = ...,
        representation: Union[L["rpy/xyz", "rpy/zyx", "eul", "exp"], None] = None,
    ) -> NDArray:  # pragma no cover
        ...

    def jacob0_dot(
        self,
        q,
        qd: ArrayLike,
        J0=None,
        representation: Union[L["rpy/xyz", "rpy/zyx", "eul", "exp"], None] = None,
    ):
        r"""
        Derivative of Jacobian

        ``robot.jacob_dot(q, qd)`` computes the rate of change of the
        Jacobian elements

        .. math::

            \dmat{J} = \frac{d \mat{J}}{d \vec{q}} \frac{d \vec{q}}{dt}

        where the first term is the rank-3 Hessian.

        Parameters
        ----------
        q
        The joint configuration of the robot
        qd
            The joint velocity of the robot
        J0
            Jacobian in {0} frame
        representation
            angular representation

        Returns
        -------
        jdot
            The derivative of the manipulator Jacobian

        Synopsis
        --------

        If ``J0`` is already calculated for the joint
        coordinates ``q`` it can be passed in to to save computation time.

        It is computed as the mode-3 product of the Hessian tensor and the
        velocity vector.

        The derivative of an analytical Jacobian can be obtained by setting
        ``representation`` as

        |``representation``   |       Rotational representation     |
        |---------------------|-------------------------------------|
        |``'rpy/xyz'``        |   RPY angular rates in XYZ order    |
        |``'rpy/zyx'``        |   RPY angular rates in XYZ order    |
        |``'eul'``            |   Euler angular rates in ZYZ order  |
        |``'exp'``            |   exponential coordinate rates      |


        References
        ----------
        - Kinematic Derivatives using the Elementary Transform
            Sequence, J. Haviland and P. Corke

        See Also
        --------
        :func:`jacob0`
        :func:`hessian0`

        """

        qd = np.array(qd)

        if representation is None:
            if J0 is None:
                J0 = self.jacob0(q)
            H = self.hessian0(q, J0=J0)

        else:
            # # determine analytic rotation
            # T = self.fkine(q).A
            # gamma = smb.r2x(smb.t2r(T), representation=representation)

            # # get transformation angular velocity to analytic velocity
            # Ai = smb.rotvelxform(
            #     gamma, representation=representation, inverse=True, full=True
            # )

            # # get analytic rate from joint rates
            # omega = J0[3:, :] @ qd
            # gamma_dot = Ai[3:, 3:] @ omega
            # Ai_dot = smb.rotvelxform_inv_dot(gamma, gamma_dot, full=True)
            # Ai_dot = sp.linalg.block_diag(np.zeros((3, 3)), Ai_dot)

            # Jd = Ai_dot @ J0 + Ai @ Jd

            # not actually sure this can be written in closed form

            H = smb.numhess(
                lambda q: self.jacob0_analytical(q, representation=representation), q
            )

            # Jd = Ai @ Jd

            # return Jd

        return np.tensordot(H, qd, (0, 0))

    @overload
    def jacobm(
        self,
        q: ArrayLike = ...,
        J: None = None,
        H: None = None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
    ) -> NDArray:  # pragma no cover
        ...

    @overload
    def jacobm(
        self,
        q: None = None,
        J: NDArray = ...,
        H: NDArray = ...,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
    ) -> NDArray:  # pragma no cover
        ...

    def jacobm(
        self,
        q=None,
        J=None,
        H=None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
    ) -> NDArray:
        r"""
        The manipulability Jacobian

        This measure relates the rate of change of the manipulability to the
        joint velocities of the robot. One of J or q is required. Supply J
        and H if already calculated to save computation time

        Parameters
        ----------
        q
            The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        J
            The manipulator Jacobian in any frame
        H
            The manipulator Hessian in any frame
        end
            the final link or Gripper which the Hessian represents
        start
            the first link which the Hessian represents

        Returns
        -------
        jacobm
            The manipulability Jacobian

        Synopsis
        --------
        Yoshikawa's manipulability measure

        .. math::

            m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

        This method returns its Jacobian with respect to configuration

        .. math::

            \frac{\partial m(\vec{q})}{\partial \vec{q}}

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """  # noqa

        end, start, _ = self._get_limit_links(end, start)

        #
        if not isinstance(axes, list):
            if axes.startswith("all"):
                axes = [True, True, True, True, True, True]
            elif axes.startswith("trans"):
                axes = [True, True, True, False, False, False]
            elif axes.startswith("rot"):
                axes = [False, False, False, True, True, True]
            else:
                raise ValueError("axes must be all, trans or rot")

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        n = J.shape[1]

        if H is None:
            H = self.hessian0(J0=J, start=start, end=end)
        else:
            verifymatrix(H, (6, self.n, self.n))

        manipulability = self.manipulability(
            q, J=J, start=start, end=end, axes=axes  # type: ignore
        )

        J = J[axes, :]  # type: ignore
        H = H[:, axes, :]  # type: ignore

        b = np.linalg.inv(J @ np.transpose(J))
        Jm = np.zeros((n, 1))

        for i in range(n):
            c = J @ np.transpose(H[i, :, :])
            Jm[i, 0] = manipulability * np.transpose(c.flatten("F")) @ b.flatten("F")

        return Jm

    # --------------------------------------------------------------------- #
    # --------- Collision Methods ----------------------------------------- #
    # --------------------------------------------------------------------- #

    def closest_point(
        self, q: ArrayLike, shape: Shape, inf_dist: float = 1.0, skip: bool = False
    ) -> Tuple[Union[int, None], Union[NDArray, None], Union[NDArray, None],]:
        """
        Find the closest point between robot and shape

        ``closest_point(shape, inf_dist)`` returns the minimum euclidean
        distance between this robot and shape, provided it is less than
        inf_dist. It will also return the points on self and shape in the
        world frame which connect the line of length distance between the
        shapes. If the distance is negative then the shapes are collided.

        Attributes
        ----------
        shape
            The shape to compare distance to
        inf_dist
            The minimum distance within which to consider
            the shape
        skip
            Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time

        Returns
        -------
        d
            distance between the robot and shape
        p1
            [x, y, z] point on the robot (in the world frame)
        p2
            [x, y, z] point on the shape (in the world frame)

        """

        if not skip:
            self._update_link_tf(q)
            self._propogate_scene_tree()
            shape._propogate_scene_tree()

        d = 10000
        p1 = None
        p2 = None

        for link in self.links:
            td, tp1, tp2 = link.closest_point(shape, inf_dist, skip=True)

            if td is not None and td < d:
                d = td
                p1 = tp1
                p2 = tp2

        if d == 10000:
            d = None

        return d, p1, p2

    def iscollided(self, q, shape: Shape, skip: bool = False) -> bool:
        """
        Check if the robot is in collision with a shape

        ``iscollided(shape)`` checks if this robot and shape have collided

        Attributes
        ----------
        shape
            The shape to compare distance to
        skip
            Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time

        Returns
        -------
        iscollided
            True if shapes have collided

        """

        if not skip:
            self._update_link_tf(q)
            self._propogate_scene_tree()
            shape._propogate_scene_tree()

        for link in self.links:
            if link.iscollided(shape, skip=True):
                return True

        if isinstance(self, rtb.Robot):
            for gripper in self.grippers:
                for link in gripper.links:
                    if link.iscollided(shape, skip=True):
                        return True

        return False

    def collided(self, q, shape: Shape, skip: bool = False) -> bool:
        """
        Check if the robot is in collision with a shape

        ``collided(shape)`` checks if this robot and shape have collided

        Attributes
        ----------
        shape
            The shape to compare distance to
        skip
            Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time

        Returns
        -------
        collided
            True if shapes have collided

        """

        warn("method collided is deprecated, use iscollided instead", FutureWarning)
        return self.iscollided(q, shape, skip=skip)

    # --------------------------------------------------------------------- #
    # --------- Constraint Methods ---------------------------------------- #
    # --------------------------------------------------------------------- #

    def joint_velocity_damper(
        self,
        ps: float = 0.05,
        pi: float = 0.1,
        n: Union[int, None] = None,
        gain: float = 1.0,
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the joint velocity damper for QP motion control

        Formulates an inequality contraint which, when optimised for will
        make it impossible for the robot to run into joint limits. Requires
        the joint limits of the robot to be specified. See examples/mmc.py
        for use case

        Attributes
        ----------
        ps
            The minimum angle (in radians) in which the joint is
            allowed to approach to its limit
        pi
            The influence angle (in radians) in which the velocity
            damper becomes active
        n
            The number of joints to consider. Defaults to all joints
        gain
            The gain for the velocity damper

        Returns
        -------
        Ain
            A (6,) vector inequality contraint for an optisator
        Bin
            b (6,) vector inequality contraint for an optisator

        """

        if n is None:
            n = self.n

        Ain = np.zeros((n, n))
        Bin = np.zeros(n)

        for i in range(n):
            if self.q[i] - self.qlim[0, i] <= pi:
                Bin[i] = -gain * (((self.qlim[0, i] - self.q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if self.qlim[1, i] - self.q[i] <= pi:
                Bin[i] = gain * ((self.qlim[1, i] - self.q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        return Ain, Bin

    def link_collision_damper(
        self,
        shape: CollisionShape,
        q: ArrayLike,
        di: float = 0.3,
        ds: float = 0.05,
        xi: float = 1.0,
        end: Union[Link, None] = None,
        start: Union[Link, None] = None,
        collision_list: Union[List[Shape], None] = None,
    ):
        """
        Compute a collision constrain for QP motion control

        Formulates an inequality contraint which, when optimised for will
        make it impossible for the robot to run into a collision. Requires
        See examples/neo.py for use case

        Attributes
        ----------
        ds
            The minimum distance in which a joint is allowed to
            approach the collision object shape
        di
            The influence distance in which the velocity
            damper becomes active
        xi
            The gain for the velocity damper
        end
            The end link of the robot to consider
        start
            The start link of the robot to consider
        collision_list
            A list of shapes to consider for collision

        Returns
        -------
        Ain
            A (6,) vector inequality contraint for an optisator
        Bin
            b (6,) vector inequality contraint for an optisator

        """

        end, start, _ = self._get_limit_links(start=start, end=end)

        links, n, _ = self.get_path(start=start, end=end)

        q = np.array(q)
        j = 0
        Ain = None
        bin = None

        def indiv_calculation(link: Link, link_col: CollisionShape, q: NDArray):
            d, wTlp, wTcp = link_col.closest_point(shape, di)

            if d is not None and wTlp is not None and wTcp is not None:
                lpTcp = -wTlp + wTcp

                norm = lpTcp / d
                norm_h = np.expand_dims(
                    np.concatenate((norm, [0.0, 0.0, 0.0])), axis=0  # type: ignore
                )

                # tool = (self.fkine(q, end=link).inv() * SE3(wTlp)).A[:3, 3]

                # Je = self.jacob0(q, end=link, tool=tool)
                # Je[:3, :] = self._T[:3, :3] @ Je[:3, :]

                # n_dim = Je.shape[1]
                # dp = norm_h @ shape.v
                # l_Ain = zeros((1, self.n))

                Je = self.jacobe(q, start=start, end=link, tool=link_col.T)
                n_dim = Je.shape[1]
                dp = norm_h @ shape.v
                l_Ain = np.zeros((1, n))

                l_Ain[0, :n_dim] = 1 * norm_h @ Je
                l_bin = (xi * (d - ds) / (di - ds)) + dp
            else:  # pragma nocover
                l_Ain = None
                l_bin = None

            return l_Ain, l_bin

        for link in links:
            if link.isjoint:
                j += 1

            if collision_list is None:
                col_list = link.collision

                for c in col_list:
                    pass
            else:
                col_list = [collision_list[j - 1]]  # pragma nocover

            for link_col in col_list:
                l_Ain, l_bin = indiv_calculation(link, link_col, q)  # type: ignore

                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = np.concatenate((Ain, l_Ain))

                    if bin is None:
                        bin = np.array(l_bin)
                    else:
                        bin = np.concatenate((bin, l_bin))

        return Ain, bin

    def vision_collision_damper(
        self,
        shape: CollisionShape,
        camera: Union["Robot", SE3, None] = None,
        camera_n: int = 0,
        q=None,
        di=0.3,
        ds=0.05,
        xi=1.0,
        end=None,
        start=None,
        collision_list=None,
    ):  # pragma nocover
        """
        Compute a vision collision constrain for QP motion control

        Formulates an inequality contraint which, when optimised for will
        make it impossible for the robot to run into a line of sight.
        See examples/fetch_vision.py for use case

        Attributes
        ----------
        camera
            The camera link, either as a robotic link or SE3
            pose
        camera_n
            Degrees of freedom of the camera link
        ds
            The minimum distance in which a joint is allowed to
            approach the collision object shape
        di
            The influence distance in which the velocity
            damper becomes active
        xi
            The gain for the velocity damper
        end
            The end link of the robot to consider
        start
            The start link of the robot to consider
        collision_list
            A list of shapes to consider for collision

        Returns
        -------
        Ain
            A (6,) vector inequality contraint for an optisator
        Bin
            b (6,) vector inequality contraint for an optisator
        """

        end, start, _ = self._get_limit_links(start=start, end=end)

        links, n, _ = self.get_path(start=start, end=end)

        q = np.array(q)
        j = 0
        Ain = None
        bin = None

        def rotation_between_vectors(a, b):
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)

            angle = np.arccos(np.dot(a, b))
            axis = np.cross(a, b)

            return SE3.AngleAxis(angle, axis)

        if isinstance(camera, rtb.BaseRobot):
            wTcp = camera.fkine(camera.q).A[:3, 3]
        elif isinstance(camera, SE3):
            wTcp = camera.t
        else:
            raise TypeError("Camera must be a robotic link or SE3 pose")

        wTtp = shape.T[:3, -1]

        # Create line of sight object
        los_mid = SE3((wTcp + wTtp) / 2)
        los_orientation = rotation_between_vectors(
            np.array([0.0, 0.0, 1.0]), wTcp - wTtp  # type: ignore
        )

        los = Cylinder(
            radius=0.001,
            length=np.linalg.norm(wTcp - wTtp),  # type: ignore
            base=(los_mid * los_orientation),
        )

        def indiv_calculation(link: Link, link_col: CollisionShape, q: NDArray):
            d, wTlp, wTvp = link_col.closest_point(los, di)

            if d is not None and wTlp is not None and wTvp is not None:
                lpTvp = -wTlp + wTvp

                norm = lpTvp / d
                norm_h = np.expand_dims(
                    np.concatenate((norm, [0.0, 0.0, 0.0])), axis=0
                )  # type: ignore

                tool = SE3(
                    (np.linalg.inv(self.fkine(q, end=link).A) @ SE3(wTlp).A)[:3, 3]
                )

                Je = self.jacob0(q, end=link, tool=tool.A)
                Je[:3, :] = self._T[:3, :3] @ Je[:3, :]
                n_dim = Je.shape[1]

                if isinstance(camera, "Robot"):
                    Jv = camera.jacob0(camera.q)
                    Jv[:3, :] = self._T[:3, :3] @ Jv[:3, :]

                    Jv *= (
                        np.linalg.norm(wTvp - shape.T[:3, -1]) / los.length
                    )  # type: ignore

                    dpc = norm_h @ Jv
                    dpc = np.concatenate(
                        (
                            dpc[0, :-camera_n],
                            np.zeros(self.n - (camera.n - camera_n)),
                            dpc[0, -camera_n:],
                        )
                    )
                else:
                    dpc = np.zeros((1, self.n + camera_n))

                dpt = norm_h @ shape.v
                dpt *= np.linalg.norm(wTvp - wTcp) / los.length  # type: ignore

                l_Ain = np.zeros((1, self.n + camera_n))
                l_Ain[0, :n_dim] = norm_h @ Je
                l_Ain -= dpc
                l_bin = (xi * (d - ds) / (di - ds)) + dpt
            else:
                l_Ain = None
                l_bin = None

            return l_Ain, l_bin

        for link in links:
            if link.isjoint:
                j += 1

            if collision_list is None:
                col_list = link.collision
            else:
                col_list = collision_list[j - 1]

            for link_col in col_list:
                l_Ain, l_bin = indiv_calculation(link, link_col, q)

                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = np.concatenate((Ain, l_Ain))

                    if bin is None:
                        bin = np.array(l_bin)
                    else:
                        bin = np.concatenate((bin, l_bin))

        return Ain, bin

    # --------------------------------------------------------------------- #
    # --------- Dynamics Methods ------------------------------------------ #
    # --------------------------------------------------------------------- #

    def rne(
        self,
        q: NDArray,
        qd: NDArray,
        qdd: NDArray,
        symbolic: bool = False,
        gravity: Union[ArrayLike, None] = None,
    ):
        """
        Compute inverse dynamics via recursive Newton-Euler formulation

        ``rne_dh(q, qd, qdd)`` where the arguments have shape (n,) where n is
        the number of robot joints.  The result has shape (n,).

        ``rne_dh(q, qd, qdd)`` where the arguments have shape (m,n) where n
        is the number of robot joints and where m is the number of steps in
        the joint trajectory.  The result has shape (m,n).

        ``rne_dh(p)`` where the input is a 1D array ``p`` = [q, qd, qdd] with
        shape (3n,), and the result has shape (n,).

        ``rne_dh(p)`` where the input is a 2D array ``p`` = [q, qd, qdd] with
        shape (m,3n) and the result has shape (m,n).

        Parameters
        ----------
        q
            Joint coordinates
        qd
            Joint velocity
        qdd
            Joint acceleration
        symbolic
            If True, supports symbolic expressions
        gravity
            Gravitational acceleration, defaults to attribute
            of self

        Returns
        -------
        tau
            Joint force/torques

        Notes
        -----
        - This version supports symbolic model parameters
        - Verified against MATLAB code

        """

        n = self.n

        # allocate intermediate variables
        Xup = SE3.Alloc(n)
        Xtree = SE3.Alloc(n)

        v = SpatialVelocity.Alloc(n)
        a = SpatialAcceleration.Alloc(n)
        f = SpatialForce.Alloc(n)
        I = SpatialInertia.Alloc(n)  # noqa
        s = []  # joint motion subspace

        # Handle trajectory case
        q = getmatrix(q, (None, None))
        qd = getmatrix(qd, (None, None))
        qdd = getmatrix(qdd, (None, None))
        l, _ = q.shape  # type: ignore

        if symbolic:  # pragma: nocover
            Q = np.empty((l, n), dtype="O")  # joint torque/force
        else:
            Q = np.empty((l, n))  # joint torque/force

        # TODO Should the dynamic parameters of static links preceding joint be
        # somehow merged with the joint?

        # A temp variable to handle static joints
        Ts = SE3()

        # A counter through joints
        j = 0

        for k in range(l):
            qk = q[k, :]
            qdk = qd[k, :]
            qddk = qdd[k, :]

            # initialize intermediate variables
            for link in self.links:
                if link.isjoint:
                    I[j] = SpatialInertia(m=link.m, r=link.r)
                    if symbolic and link.Ts is None:  # pragma: nocover
                        Xtree[j] = SE3(np.eye(4, dtype="O"), check=False)
                    elif link.Ts is not None:
                        Xtree[j] = Ts * SE3(link.Ts, check=False)

                    if link.v is not None:
                        s.append(link.v.s)

                    # Increment the joint counter
                    j += 1

                    # Reset the Ts tracker
                    Ts = SE3()
                else:  # pragma nocover
                    # TODO Keep track of inertia and transform???
                    if link.Ts is not None:
                        Ts *= SE3(link.Ts, check=False)

            if gravity is None:
                a_grav = -SpatialAcceleration(self.gravity)
            else:  # pragma nocover
                a_grav = -SpatialAcceleration(gravity)

            # forward recursion
            for j in range(0, n):
                vJ = SpatialVelocity(s[j] * qdk[j])

                # transform from parent(j) to j
                Xup[j] = SE3(self.links[j].A(qk[j])).inv()

                if self.links[j].parent is None:
                    v[j] = vJ
                    a[j] = Xup[j] * a_grav + SpatialAcceleration(s[j] * qddk[j])
                else:
                    jp = self.links[j].parent.jindex  # type: ignore
                    v[j] = Xup[j] * v[jp] + vJ
                    a[j] = (
                        Xup[j] * a[jp] + SpatialAcceleration(s[j] * qddk[j]) + v[j] @ vJ
                    )

                f[j] = I[j] * a[j] + v[j] @ (I[j] * v[j])

            # backward recursion
            for j in reversed(range(0, n)):
                # next line could be dot(), but fails for symbolic arguments
                Q[k, j] = sum(f[j].A * s[j])

                if self.links[j].parent is not None:
                    jp = self.links[j].parent.jindex  # type: ignore
                    f[jp] = f[jp] + Xup[j] * f[j]

        if l == 1:
            return Q[0]
        else:  # pragma nocover
            return Q


# ============================================================================= #
# ================= Robot2 Class ============================================== #
# ============================================================================= #


class Robot2(BaseRobot[Link2]):
    def __init__(self, arg, **kwargs):
        if isinstance(arg, ETS2):
            # we're passed an ETS string
            links = []
            # chop it up into segments, a link frame after every joint
            parent = None
            for j, ets_j in enumerate(arg.split()):
                elink = Link2(ETS2(ets_j), parent=parent, name=f"link{j:d}")
                parent = elink
                if (
                    elink.qlim is None
                    and elink.v is not None
                    and elink.v.qlim is not None
                ):  # pragma nocover
                    elink.qlim = elink.v.qlim
                links.append(elink)

        elif smb.islistof(arg, Link2):
            links = arg

        else:  # pragma nocover
            raise TypeError("constructor argument must be ETS2 or list of Link2")

        super().__init__(links, **kwargs)

        # Should just set it to None
        self.base = SE2()  # override superclass

    @property
    def base(self) -> SE2:
        """
        Get/set robot base transform (Robot superclass)

        ``robot.base`` is the robot base transform

        Returns
        -------
        base
            robot tool transform

        - ``robot.base = ...`` checks and sets the robot base transform

        Notes
        -----
        - The private attribute ``_base`` will be None in the case of
            no base transform, but this property will return ``SE3()`` which
            is an identity matrix.
        """
        if self._base is None:  # pragma nocover
            self._base = SE2()

        # return a copy, otherwise somebody with
        # reference to the base can change it
        return self._base.copy()

    @base.setter
    def base(self, T):
        if isinstance(T, SE2):
            self._base = T
        elif SE2.isvalid(T):  # pragma nocover
            self._tool = SE2(T, check=True)

    def jacob0(self, q, start=None, end=None):
        return self.ets(start, end).jacob0(q)

    def jacobe(self, q, start=None, end=None):
        return self.ets(start, end).jacobe(q)

    def fkine(self, q, end=None, start=None):
        return self.ets(start, end).fkine(q)

    @property
    def reach(self) -> float:
        r"""
        Reach of the robot

        A conservative estimate of the reach of the robot. It is computed as
        the sum of the translational ETs that define the link transform.

        Note
        ----
        Computed on the first access. If kinematic parameters
        subsequently change this will not be reflected.

        Returns
        -------
        reach
            Maximum reach of the robot

        Notes
        -----
        - Probably an overestimate of reach
        - Used by numerical inverse kinematics to scale translational
          error.
        - For a prismatic joint, uses ``qlim`` if it is set

        """

        # TODO
        # This should be a start, end method and compute the reach based on the
        # given ets. Then use an lru_cache to speed up return

        if self._reach is None:
            d_all = []
            for link in self.ee_links:
                d = 0
                while True:
                    for et in link.ets:
                        if et.istranslation:
                            if et.isjoint:
                                # the length of a prismatic joint depends on the
                                # joint limits.  They might be set in the ET
                                # or in the Link depending on how the robot
                                # was constructed
                                if link.qlim is not None:
                                    d += max(link.qlim)
                                elif et.qlim is not None:  # pragma nocover
                                    d += max(et.qlim)
                            else:
                                d += abs(et.eta)
                    link = link.parent
                    if link is None or isinstance(link, str):
                        d_all.append(d)
                        break

            self._reach = max(d_all)
        return self._reach

    def fkine_all(self, q: ArrayLike) -> SE2:
        """
        Compute the pose of every link frame

        ``T = robot.fkine_all(q)`` is  an SE3 instance with ``robot.nlinks +
        1`` values:

        - ``T[0]`` is the base transform
        - ``T[i]`` is the pose of link whose ``number`` is ``i``

        Parameters
        ----------
        q
            The joint configuration

        Returns
        -------
        fkine_all
            Pose of all links

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """  # noqa

        q = getvector(q)

        Tbase = SE2(self.base)  # add base, also sets the type

        linkframes = Tbase.__class__.Alloc(self.nlinks + 1)
        linkframes[0] = Tbase

        def recurse(Tall, Tparent, q, link):
            # if joint??
            T = Tparent
            while True:
                T *= SE2(link.A(q[link.jindex]))

                Tall[link.number] = T

                if link.nchildren == 0:
                    # no children
                    return
                elif link.nchildren == 1:
                    # one child
                    if link in self.ee_links:  # pragma nocover
                        # this link is an end-effector, go no further
                        return
                    link = link.children[0]
                    continue
                else:
                    # multiple children
                    for child in link.children:
                        recurse(Tall, T, q, child)
                    return

        recurse(linkframes, Tbase, q, self.links[0])

        return linkframes
