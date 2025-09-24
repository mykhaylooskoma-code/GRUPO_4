import numpy as np
from src.ndwi import compute_ndwi

def test_same_input_same_output():
    g = np.array([[0,1],[2,3.]], float); n = g.copy()
    o1 = compute_ndwi(g,n); o2 = compute_ndwi(g,n)
    assert np.allclose(o1,o2, equal_nan=True)

def test_division_by_zero_nan():
    g = np.array([[1.0,0.0]]); n = np.array([[-1.0,0.0]])
    out = compute_ndwi(g,n, eps=1e-12)
    assert np.isnan(out).all()

def test_mask_and_nan():
    g = np.array([[np.nan,0.5]]); n = np.array([[0.2,0.1]]); m = np.array([[True,False]])
    out = compute_ndwi(g,n,mask=m)
    assert np.isnan(out[0,0]) and np.isnan(out[0,1])

def test_clip_range():
    g = np.array([[1000.0]]); n = np.array([[0.0]])
    out = compute_ndwi(g,n,clip=True)
    assert out.min() >= -1 and out.max() <= 1
