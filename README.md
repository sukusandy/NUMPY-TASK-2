# NUMPY-TASK-2
#### 1. Import the numpy package under the name `np` (★☆☆) 
(**hint**: import … as …)

import numpy as np

#### 2. Print the numpy version and the configuration (★☆☆) 
(**hint**: np.\_\_version\_\_, np.show\_config)

print (np.__version__)

print(np.show_config())

#### 3. Create a null vector of size 10 (★☆☆) 
(**hint**: np.zeros)

print(np.zeros(10))

Z=np.zeros(10)
print(Z)

#### 4.  How to find the memory size of any array (★☆☆) 
(**hint**: size, itemsize)

x = np.array([100,20,34])
print(x.size)

#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆) 
(**hint**: np.info)

print(np.info(np.add))

#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 
(**hint**: array\[4\])

x=np.zeros(10)
x[4]=1
print(x)

#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) 
(**hint**: np.arange)

x= np.arange(10,49)
print(x)

#### 8.  Reverse a vector (first element becomes last) (★☆☆) 
(**hint**: array\[::-1\])

x = x[::-1]
print(x)

#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 
(**hint**: reshape)

x =  np.arange(0, 9).reshape(3,3)
print(x)

#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 
(**hint**: np.nonzero)

import numpy as np
print (np.nonzero([1,2,0,0,4,0]))

#### 11. Create a 3x3 identity matrix (★☆☆) 
(**hint**: np.eye)

x = np.eye(3)
print(x)

#### 12. Create a 3x3x3 array with random values (★☆☆) 
(**hint**: np.random.random)

x = np.random.random((10,10,10))
print(x)

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 
(**hint**: min, max)

x = np.random.random((10,10))
xmin, xmax = x.min(), x.max()
print(xmin, xmax)

#### 14. Create a random vector of size 30 and find the mean value (★☆☆) 
(**hint**: mean)

f = np.random.random((30))
np.mean(f)

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
(**hint**: array\[1:-1, 1:-1\])

g = np.ones((10,5))

g[1:-1, 1:-1] = 0
print(g)

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) 
(**hint**: np.pad)

x = np.ones((3,3))
x = np.pad(x, pad_width=1, mode='constant', constant_values=0)
print(x)

#### 17. What is the result of the following expression? (★☆☆) 
(**hint**: NaN = not a number, inf = infinity)

```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```

print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 
(**hint**: np.diag)

x = np.diag([1, 2, 3, 4,5])
print(x)

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 
(**hint**: array\[::2\])

x = np.zeros((8,8),dtype=int)
x[1::2,::2] = 1
print(x)

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 
(**hint**: np.unravel_index)

print (np.unravel_index(100, (6,7,8)))

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) 
(**hint**: np.tile)

array= np.array([[0,1], [1,0]])
Z = np.tile(array,(4,4))
print (Z)

#### 22. Normalize a 5x5 random matrix (★☆☆) 
(**hint**: (x - min) / (max - min))

Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z= (Z-Zmin)/(Zmax-Zmin)
print (Z)

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) 
(**hint**: np.dtype)

color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 
(**hint**: np.dot | @)

Z= np.dot(np.ones((5,3)), np.ones((3,2)))
print (Z)

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) 
(**hint**: >, <=)

Z = np.arange(15)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)

#### 26. What is the output of the following script? (★☆☆) 
(**hint**: np.sum)



print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z

#### 28. What are the result of the following expressions?

```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)

#### 29. How to round away from zero a float array ? (★☆☆) 
(**hint**: np.uniform, np.copysign, np.ceil, np.abs)

a = np.random.uniform(-12, +10, 10)
np.copysign(np.ceil(np.abs(a)), a)

#### 30. How to find common values between two arrays? (★☆☆) 
(**hint**: np.intersect1d)

x= np.arange(0, 10)
y = np.arange(5, 15)
np.intersect1d(x, y)

#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) 
(**hint**: np.seterr, np.errstate)

defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

#### 32. Is the following expressions true? (★☆☆) 
(**hint**: imaginary number)

```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

print(np.sqrt(-1) == np.emath.sqrt(-1))


#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) 
(**hint**: np.datetime64, np.timedelta64)

todays_date = np.datetime64('today', 'D')
print("Today's Date: ", todays_date)

yesterdays_date = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print("Yesterday's Date: ", yesterdays_date)

tomorrows_date = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Tomorrow's Date: ", tomorrows_date)

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) 
(**hint**: np.arange(dtype=datetime64\['D'\]))

x = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(x)

#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) 
(**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))

A = np.ones(3) * 1
B = np.ones(3) * 2
C = np.ones(3) * 3
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)

#### 36. Extract the integer part of a random array using 5 different methods (★★☆) 
(**hint**: %, np.floor, np.ceil, astype, np.trunc)

x = np.random.uniform(-10, +10, 10)
x.astype(np.int32)
np.trunc(x)

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 
(**hint**: np.arange)

x = np.zeros((5, 5))
x += np.arange(5)
print(x)

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) 
(**hint**: np.fromiter)

def generate_integers():
    for i in range(10):
        yield i
np.fromiter(generate_integers(), dtype=np.float32, count=-1)

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 
(**hint**: np.linspace)

np.linspace(start=0, stop=11, num=10, endpoint=False)

#### 40. Create a random vector of size 10 and sort it (★★☆) 
(**hint**: sort)

x = np.random.random(size=10)
x.sort()
print(x)

#### 41. How to sum a small array faster than np.sum? (★★☆) 
(**hint**: np.add.reduce)

import time

begin_time = time.time()
x = np.arange(1000000)
np.sum(x)
end_time = time.time()
print(str(1000*(end_time - begin_time)))

begin_time2 = time.time()
x = np.arange(1000000)
np.add.reduce(x)
end_time2 = time.time()
print(str(1000*(end_time2 - begin_time2)))

#### 42. Consider two random array A and B, check if they are equal (★★☆) 
(**hint**: np.allclose, np.array\_equal)

A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)

#### 43. Make an array immutable (read-only) (★★☆) 
(**hint**: flags.writeable)

Z = np.zeros(10)
Z.flags.writeable = False

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) 
(**hint**: np.sqrt, np.arctan2)

Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 
(**hint**: argmax)

Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)

#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) 
(**hint**: np.meshgrid)

Z = np.zeros((10,10), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
                             np.linspace(0,1,10))
print(Z)

####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) 
(**hint**: np.subtract.outer)

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 
(**hint**: np.iinfo, np.finfo, eps)

for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

#### 49. How to print all the values of an array? (★★☆) 
(**hint**: np.set\_printoptions)

np.set_printoptions(threshold=float("inf"))
Z = np.zeros((40,40))
print(Z)

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) 
(**hint**: argmin)

Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) 
(**hint**: dtype)

Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                   ('y', float, 1)]),
                    ('color',    [ ('r', float, 1),
                                   ('g', float, 1),
                                   ('b', float, 1)])])
print(Z)

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) 
(**hint**: np.atleast\_2d, T, np.sqrt)

Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? 
(**hint**: astype(copy=False))

Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)

#### 54. How to read the following file? (★★☆) 
(**hint**: np.genfromtxt)

```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

from io import StringIO
Z = np.genfromtxt("missing.dat", delimiter=",")
print(Z)

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) 
(**hint**: np.ndenumerate, np.ndindex)

Z = np.genfromtxt("missing.dat", delimiter=",")

#### 56. Generate a generic 2D Gaussian-like array (★★☆) 
(**hint**: np.meshgrid, np.exp)

X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)

#### 57. How to randomly place p elements in a 2D array? (★★☆) 
(**hint**: np.put, np.random.choice)

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print (Z)

#### 58. Subtract the mean of each row of a matrix (★★☆) 
(**hint**: mean(axis=,keepdims=))


X = np.random.rand(5, 10)

# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)

# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)

#### 59. How to sort an array by the nth column? (★★☆) 
(**hint**: argsort)

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])

#### 60. How to tell if a given 2D array has null columns? (★★☆) 
(**hint**: any, ~)



#### 61. Find the nearest value from a given value in an array (★★☆) 
(**hint**: np.abs, argmin, flat)

Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) 
(**hint**: np.nditer)

A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])

#### 63. Create an array class that has a name attribute (★★☆) 
(**hint**: class method)

class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) 
(**hint**: np.bincount | np.add.at)

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) 
(**hint**: np.bincount)

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) 
(**hint**: np.unique)

w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) 
(**hint**: sum(axis=(-2,-1)))

A = np.random.randint(0,10,(3,4,3,4))
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) 
(**hint**: np.bincount)

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

#### 69. How to get the diagonal of a dot product? (★★★) 
(**hint**: np.diag)

A = np.random.randint(0,10,(3,3))
B= np.random.randint(0,10,(3,3))
#Slow version

np.diag(np.dot(A, B))

# Fast version
np.sum(A * B.T, axis=1)

# Faster version
np.einsum("ij,ji->i", A, B)

#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 
(**hint**: array\[::4\])

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 
(**hint**: array\[:, :, None\])

A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])

#### 72. How to swap two rows of an array? (★★★) 
(**hint**: array\[\[\]\] = array\[\[\]\])

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)


#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) 
(**hint**: repeat, np.roll, np.sort, view, np.unique)

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)

#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) 
(**hint**: np.repeat)

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)

#### 75. How to compute averages using a sliding window over an array? (★★★) 
(**hint**: np.cumsum)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) 
(**hint**: from numpy.lib import stride_tricks)

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) 
(**hint**: np.logical_not, np.negative)

Z = np.random.randint(0,2,100)
print ('original: ')
print (Z)
print('Negating a boolean: ')
print(np.logical_not(Z, out=Z))

Z = np.random.uniform(-1.0,1.0,10)
print ('original: ')
print (Z)
print ('Change the sign of float inplace: ')
print(np.negative(Z, out=Z))


#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

P0 = np.random.uniform(-10, 10, (5,2))
P1 = np.random.uniform(-10,10,(5,2))
p = np.random.uniform(-10, 10, (5,2))
print (np.array([distance(P0,P1,p_i) for p_i in p]))

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) 
(**hint**: minimum, maximum)

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) 
(**hint**: stride\_tricks.as\_strided)

Z = np.arange(1,15,dtype=int)

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
R = rolling(Z, 4)
print ('original: ')
print (Z)
print ('after strides: ')
print(R)

#### 82. Compute a matrix rank (★★★) 
(**hint**: np.linalg.svd) (suggestion: np.linalg.svd)

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print (rank)

#### 83. How to find the most frequent value in an array? 
(**hint**: np.bincount, argmax)

Z = np.random.randint(0,10,50)
print (Z)
print('rank:', np.bincount(Z).argmax())

Z = np.random.randint(0,5,(6,6))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = np.lib.stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 
(**hint**: stride\_tricks.as\_strided)

#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) 
(**hint**: class method)

class Symetric(np.ndarray):
    def __setitem__(self, (i,j), value):
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)

#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) 
(**hint**: np.tensordot)

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) 
(**hint**: np.add.reduceat)

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print ('input array')
print (Z)
print ('block sum')
print (S)

#### 88. How to implement the Game of Life using numpy arrays? (★★★)

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)


#### 89. How to get the n largest values of an array (★★★) 
(**hint**: np.argsort | np.argpartition)

Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
print (Z[np.argpartition(-Z,n)[:n]])

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) 
(**hint**: np.indices)

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))

#### 91. How to create a record array from a regular array? (★★★) 
(**hint**: np.core.records.fromarrays)

Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) 
(**hint**: np.power, \*, np.einsum)

x = np.random.rand(int(5e7))

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) 
(**hint**: np.where)

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows)

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

Z = np.random.randint(0,5,(10,3))
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(Z)
print(U)

#### 95. Convert a vector of ints into a matrix binary representation (★★★) 
(**hint**: np.unpackbits)

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)


I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))

#### 96. Given a two dimensional array, how to extract unique rows? (★★★) 
(**hint**: np.ascontiguousarray)

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) 
(**hint**: np.einsum)

A= np.arange(3)
B =  np.arange(12).reshape(3,4)
print (A)
#np.einsum('ii->', A)       # np.sum(A)
#np.einsum('i,i->i', A, B) # A * B
#np.einsum('i,i', A, B)    # np.inner(A, B)
#np.einsum('...i,j', A, B)    # np.outer(A, B)

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? 
(**hint**: np.cumsum, np.interp)

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) 
(**hint**: np.logical\_and.reduce, np.mod)

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) 
(**hint**: np.percentile)

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
