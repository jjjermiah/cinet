from cinet import *
import pandas as pd

def test_init_deepCINET():
    """Test initializing DeepCINET"""
    DCmodel = deepCINET()
    return DCmodel

def test_init_ECINET():
    """Test initializing ECINET (Linear CINET)"""
    ECmodel = ECINET(delta=0.05)
    
    assert (ECmodel.delta == 0.05), "Delta not initialized correctly"

def integrative_test_deepCINET(): 
    DCmodel = deepCINET(seed=420)
    train_file = r'/home/gputwo/bhklab/kevint/cinet/tests/fake_train_data.txt'
    train_df = pd.read_csv(train_file).set_index('cell_line')
    X = train_df.iloc[:,1:]
    y = train_df.iloc[:,0]

    DCmodel.fit(X,y)
    assert (DCmodel.score(X,y) == 0.7077551020408164), "DeepCINET Integrative Test did not not achieve the correct value on scoring the model against training data"

    test_file = r'/home/gputwo/bhklab/kevint/cinet/tests/fake_test_data.txt'
    test_df = pd.read_csv(test_file).set_index('cell_line')
    X2 = test_df.iloc[:,1:]
    y2 = test_df.iloc[:,0]
    assert (DCmodel.score(X2,y2) == 0.7028571428571428), "DeepCINET Integrative Test did not not achieve the correct value on scoring the model against testing data"

def integrative_test_ECINET(): 
    ECmodel = ECINET(seed=420)
    train_file = r'/home/gputwo/bhklab/kevint/cinet/tests/fake_train_data.txt'
    train_df = pd.read_csv(train_file).set_index('cell_line')
    X = train_df.iloc[:,1:]
    y = train_df.iloc[:,0]

    ECmodel.fit(X,y)
    assert (ECmodel.score(X,y) == 0.6351020408163265), "ECINET Integrative Test did not not achieve the correct value on scoring the model against training data"

    test_file = r'/home/gputwo/bhklab/kevint/cinet/tests/fake_test_data.txt'
    test_df = pd.read_csv(test_file).set_index('cell_line')
    X2 = test_df.iloc[:,1:]
    y2 = test_df.iloc[:,0]
    assert (ECmodel.score(X2,y2) == 0.6293877551020408), "ECINET Integrative Test did not not achieve the correct value on scoring the model against testing data"
