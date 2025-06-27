import numpy as np 

def test_markov_chain_static():
    Q = np.array([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]])
    s = np.array([0.6, 0.2, 0.2])
    for i in range(100):
        s_next = s @ Q
        s = s_next

    assert np.allclose(s, np.array([0.625, 0.3125, 0.0625]), atol=1e-2)
