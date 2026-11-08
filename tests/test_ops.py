import numpy as np
import pytest

from numpy_kit.ops import (
  pairwise,
  pairwise_loop,
  pairwise_v2,
  softmax,
  softmax_loop,
  softmax_loop_1d_loop,
  standardize,
  standardize_loop,
)


@pytest.fixture
def rng():
  return np.random.default_rng(42)

# Softmax

def test_softmax_loop_sums_to_one():
  out = softmax_loop_1d_loop(np.array([1.0, 2.0, 3.0]))
  assert len(out) == 3
  assert abs(sum(out) - 1) < 1e-9
  assert not np.isnan(softmax_loop_1d_loop(np.array([1000.0, 1001.0]))).any()


def test_empty_array():
  with pytest.raises(ValueError, match = 'empty'):
    softmax_loop_1d_loop(np.array([]))

def test_different_array_sizes():
  arr1d = softmax_loop(np.zeros(10))
  assert np.allclose(arr1d, 0.1)
  with pytest.raises(ValueError, match='1D or 2D array'):
    softmax_loop(np.array([[[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]]))



def test_softmax_loop_2d():
  out = softmax_loop(np.array([[1.0, 1.0, 1.0],[1.0,2.0,3.0]]))
  assert out.shape == (2,3)
  assert np.allclose(out.sum(axis=1), 1)
  #assert not np.isnan(softmax_loop(np.array([1000.0, 1001.0]))).any()

def test_array_not_altered():
  x = np.array([[1, 2, 3]])
  x_copy = x.copy()
  softmax_loop(x)
  assert np.array_equal(x, x_copy)   # input must be unchanged

def test_softmax(rng):
  out = softmax(np.array([1.0, 2.0, 3.0]))
  assert len(out) == 3
  assert abs(sum(out) - 1) < 1e-9
  assert not np.isnan(softmax(np.array([1000.0, 1001.0]))).any()
  out2 = np.array([1.0, 2.0, 3.0])
  assert np.allclose(softmax(out2), softmax(out2+100))

@pytest.mark.parametrize("shape", [(3,),(3,4),(4,4),(5,2)])

def test_softmax_and_softmax_loop(shape, rng):
  
  arr = rng.standard_normal(shape)
  assert np.allclose(softmax(arr), softmax_loop(arr))

# ----------------------------------------------
# Pairwise Dsiatnce

def test_pairwise_loop():
  a = np.array([[0,0], [1,1]])
  b = np.array([[1,0], [0,2], [3,4]])
  a1 = np.zeros((4,3))
  b1 = np.zeros((6,3))
  assert np.allclose(pairwise_loop(a,b), np.array([[1.0, 2.0, 5.0], [1.0, np.sqrt(2), np.sqrt(13)]]))
  assert pairwise_loop(a1,b1).shape == (4,6)


def test_loop_different_coordinates():
  a = np.zeros((4,5))
  b = np.zeros((4,3))
  with pytest.raises(ValueError, match = 'number of coordinates'):
    pairwise_loop(a,b)
  
  a = np.zeros((4,5,5))
  b = np.zeros((4,3,4))
  with pytest.raises(ValueError, match = '2D arrays'):
    pairwise_loop(a,b)

def test_no_negatives(rng):
  a = rng.standard_normal((2,3))
  b = rng.standard_normal((4,3))
  d = pairwise_loop(a,b)
  assert (d >=0).all()

def test_distance_from_itself(rng):
  a = rng.standard_normal((5,3))
  d = pairwise_loop(a,a)
  assert np.allclose(np.diag(d), 0)
  assert np.allclose(d,d.T)


def test_pairwise():
  a = np.array([[0,0], [1,1]])
  b = np.array([[1,0], [0,2], [3,4]])
  a1 = np.zeros((4,3))
  b1 = np.zeros((6,3))
  assert np.allclose(pairwise(a,b), np.array([[1.0, 2.0, 5.0], [1.0, np.sqrt(2), np.sqrt(13)]]))
  assert pairwise(a1,b1).shape == (4,6)


@pytest.mark.parametrize("shape_a, shape_b", [((2, 3), (4, 3)), ((5, 2), (5, 2)), ((1, 8), (6, 8))])
def test_pairwise_matches_loop(shape_a, shape_b, rng):
    a = rng.standard_normal(shape_a)   
    b = rng.standard_normal(shape_b)
    assert np.allclose(pairwise(a,b), pairwise_loop(a,b))

def test_different_coordinates():
  a = np.zeros((4,5))
  b = np.zeros((4,3))
  with pytest.raises(ValueError, match = 'number of coordinates'):
    pairwise(a,b)
  
  a = np.zeros((4,5,5))
  b = np.zeros((4,3,4))
  with pytest.raises(ValueError, match = '2D arrays'):
    pairwise(a,b)


@pytest.mark.parametrize("shape_a, shape_b", [((2, 3), (4, 3)), ((5, 2), (5, 2)), ((1, 8), (6, 8))])
def test_pairwise_v2_matches_loop(shape_a, shape_b, rng):
    a = rng.standard_normal(shape_a)   
    b = rng.standard_normal(shape_b)
    assert np.allclose(pairwise_v2(a,b), pairwise_loop(a,b))

def test_pairwise_v2_distance_from_itself(rng):
  a = rng.standard_normal((200,64))
  d = pairwise_v2(a,a)
  assert not np.isnan(d).any()
  assert np.allclose(np.diag(d), 0, atol = 1e-6)

def test_pairwise_v2_different_coordinates():
  a = np.zeros((4,5))
  b = np.zeros((4,3))
  with pytest.raises(ValueError, match = 'number of coordinates'):
    pairwise_v2(a,b)
  
  a = np.zeros((4,5,5))
  b = np.zeros((4,3,4))
  with pytest.raises(ValueError, match = '2D arrays'):
    pairwise_v2(a,b)

# ---------------------------------
# Standardize

def test_standardize_loop():
  x = np.array([[1,10], [2,10], [3,10]])
  y = x.copy()

  z = np.array([[-np.sqrt(1.5), 0], [0,0], [np.sqrt(1.5), 0]])

  mean = [2,10]
  stds = [np.sqrt(2/3),0]

  zl, meanl, stdsl = standardize_loop(x)
  
  assert np.allclose(z, zl) 
  assert np.allclose(stds, stdsl) 
  assert np.allclose(mean, meanl)
  assert np.allclose(x,y)

def test_mean_std_in_loop(rng):
  x = rng.standard_normal((50,7))

  z, meanl, stdl = standardize_loop(x)

  assert np.allclose(x.mean(axis = 0), meanl)
  assert np.allclose(x.std(axis = 0), stdl)
  assert np.allclose(z.mean(axis = 0), 0)
  assert np.allclose(z.std(axis = 0), 1)

def test_constant_columns_loop(rng):
  x = rng.standard_normal((50,4))
  x[:,1] = 0.1

  z,_,_ = standardize_loop(x)

  assert not np.isnan(z).any()
  assert np.allclose(z[:,1], 0)



def test_different_coordinates_standardize_loop():
  a = np.zeros(4)
  with pytest.raises(ValueError, match = '2D array'):
    standardize_loop(a)
  
  a = np.zeros((4,5,5))
  with pytest.raises(ValueError, match = '2D array'):
    standardize_loop(a)


@pytest.mark.parametrize("shape", [(5, 3), (50, 7), (1, 4)])
def test_standardize_matches_loop(shape, rng):
    a = rng.standard_normal(shape)
    zl, meanl, stdl = standardize_loop(a)
    z, means, stds = standardize(a)    
    assert np.allclose(zl, z)
    assert np.allclose(meanl, means)
    assert np.allclose(stdl, stds)


def test_constant_columns(rng):
  x = rng.standard_normal((50,4))
  x[:,1] = 0.1

  z,_,_ = standardize(x)

  assert not np.isnan(z).any()
  assert np.allclose(z[:,1], 0)


def test_different_coordinates_standardize():
  a = np.zeros(4)
  with pytest.raises(ValueError, match = '2D array'):
    standardize(a)
  
  a = np.zeros((4,5,5))
  with pytest.raises(ValueError, match = '2D array'):
    standardize(a)

  







  






