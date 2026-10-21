"""Visual geometry of a URDF robot reaches the links, or fails at parse time"""

import pytest

from roboticstoolbox.models.URDF.URDFRobot import _links_from_urdf
from roboticstoolbox.tools.urdf import URDF, URDFError

_LINK = """<robot name="r"><link name="base">
  <inertial><mass value="1"/>
    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  {visuals}
  <collision><geometry><box size="1 1 1"/></geometry></collision>
</link></robot>"""


def test_visual_shapes_are_set_on_the_link() -> None:
    urdf = URDF.loadstr(
        _LINK.format(
            visuals=(
                '<visual><geometry><box size="1 1 1"/></geometry></visual>'
                '<visual><geometry><sphere radius="0.1"/></geometry></visual>'
            )
        ),
        None,
    )
    links, _ = _links_from_urdf(urdf)
    assert len(links[0].geometry.scene_children) == 2


@pytest.mark.parametrize(
    "visual",
    [
        "<visual><geometry/></visual>",
        "<visual/>",
        "<visual><geometry><pyramid/></geometry></visual>",
        '<visual><geometry><mesh filename="package://nope/missing.stl"/>'
        "</geometry></visual>",
    ],
)
def test_a_visual_without_a_shape_is_rejected_when_parsed(visual: str) -> None:
    # so that _links_from_urdf never meets a visual without an `ob`
    with pytest.raises(URDFError):
        URDF.loadstr(_LINK.format(visuals=visual), None)
