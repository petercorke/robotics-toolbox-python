    import numpy.testing as nt
    import unittest
    
    from transforms import *
            
    class TestUnitQuaternion(unittest.TestCase):
        
        def test_constructor(self):
            nt.assert_array_almost_equal(UnitQuaternion().vec, np.r_[1,0,0,0])
            
            nt.assert_array_almost_equal(UnitQuaternion.Rx(90,'deg').vec, np.r_[1,1,0,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Rx(-90,'deg').vec, np.r_[1,-1,0,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Ry(90,'deg').vec, np.r_[1,0,1,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Ry(-90,'deg').vec, np.r_[1,0,-1,0]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Rz(90,'deg').vec, np.r_[1,0,0,1]/math.sqrt(2))
            nt.assert_array_almost_equal(UnitQuaternion.Rz(-90,'deg').vec, np.r_[1,0,0,-1]/math.sqrt(2))
            
    unittest.main()
