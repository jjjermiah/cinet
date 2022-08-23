from cinet import *

def test_init_deepCINET():
    """Test initializing DeepCINET"""
    DCmodel = deepCINET()
    return DCmodel

def test_init_ECINET():
    """Test initializing ECINET (Linear CINET)"""
    ECmodel = ECINET(delta=0.05)
    
    assert (ECmodel.delta == 0.05), "Delta not initialized correctly"