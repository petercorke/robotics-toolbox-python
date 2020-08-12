import numpy as np
import pytest

from ropy.backend import URDF, Link, Joint, Transmission, Material


def test_urdfpy(tmpdir):
    outfn = tmpdir.mkdir('urdf').join('ur5.urdf').strpath

    # Load
    u = URDF.load('ropy/backend/urdf/tests/data/ur5.urdf')

    assert isinstance(u, URDF)
    for j in u.joints:
        assert isinstance(j, Joint)
    for l in u.links:
        assert isinstance(l, Link)
    for t in u.transmissions:
        assert isinstance(t, Transmission)
    for m in u.materials:
        assert isinstance(m, Material)

    # Test fk
    fk = u.link_fk()
    assert isinstance(fk, dict)
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4, 4)

    fk = u.link_fk({'shoulder_pan_joint': 2.0})
    assert isinstance(fk, dict)
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4, 4)

    fk = u.link_fk(np.zeros(6))
    assert isinstance(fk, dict)
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4, 4)

    fk = u.link_fk(np.zeros(6), link='upper_arm_link')
    assert isinstance(fk, np.ndarray)
    assert fk.shape == (4, 4)

    fk = u.link_fk(links=['shoulder_link', 'upper_arm_link'])
    assert isinstance(fk, dict)
    assert len(fk) == 2
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4, 4)

    fk = u.link_fk(links=list(u.links)[:2])
    assert isinstance(fk, dict)
    assert len(fk) == 2
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4, 4)

    cfg = {j.name: np.random.uniform(size=1000) for j in u.actuated_joints}
    fk = u.link_fk_batch(cfgs=cfg)
    for key in fk:
        assert isinstance(fk[key], np.ndarray)
        assert fk[key].shape == (1000, 4, 4)

    # Test save
    u.save(outfn)

    nu = URDF.load(outfn)
    assert len(u.links) == len(nu.links)
    assert len(u.joints) == len(nu.joints)

    # Test join
    with pytest.raises(ValueError):
        x = u.join(u, link=u.link_map['tool0'])
    x = u.join(u, link=u.link_map['tool0'], name='copy', prefix='prefix')
    assert isinstance(x, URDF)
    assert x.name == 'copy'
    assert len(x.joints) == 2 * len(u.joints) + 1
    assert len(x.links) == 2 * len(u.links)

    # Test scale
    x = u.copy(scale=3)
    assert isinstance(x, URDF)
    x = x.copy(scale=[1, 1, 3])
    assert isinstance(x, URDF)
