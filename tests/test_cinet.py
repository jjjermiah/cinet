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

def test_integrative_deepCINET(): 
    DCmodel = deepCINET(seed=420)
    train_file = r'tests/fake_train_data.txt'
    train_df = pd.read_csv(train_file).set_index('cell_line')
    X = train_df.iloc[:,1:]
    y = train_df.iloc[:,0]

    DCmodel.fit(X,y)
    result = DCmodel.score(X,y)
    print(result)
    assert (round(result, 1) == 0.7), "DeepCINET Integrative Test did not not achieve the correct value on scoring the model against training data"

    test_file = r'tests/fake_test_data.txt'
    test_df = pd.read_csv(test_file).set_index('cell_line')
    X2 = test_df.iloc[:,1:]
    y2 = test_df.iloc[:,0]
    result2 = DCmodel.score(X2,y2)
    assert (round(result2, 1) == 0.7), "DeepCINET Integrative Test did not not achieve the correct value on scoring the model against testing data"

def test_integrative_ECINET(): 
    ECmodel = ECINET(seed=420)
    train_file = r'tests/fake_train_data.txt'
    train_df = pd.read_csv(train_file).set_index('cell_line')
    X = train_df.iloc[:,1:]
    y = train_df.iloc[:,0]

    ECmodel.fit(X,y)
    result = ECmodel.score(X,y)
    print(result)
    assert (round(result, 1) == 0.6), "ECINET Integrative Test did not not achieve the correct value on scoring the model against training data"

    test_file = r'tests/fake_test_data.txt'
    test_df = pd.read_csv(test_file).set_index('cell_line')
    X2 = test_df.iloc[:,1:]
    y2 = test_df.iloc[:,0]
    result2 = ECmodel.score(X2,y2)
    assert (round(result2, 1) == 0.6), "ECINET Integrative Test did not not achieve the correct value on scoring the model against testing data"
