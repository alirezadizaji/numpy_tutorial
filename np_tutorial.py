import numpy as np
##################### Arrays ########################
#       shape -> tuple giving size of array         #
a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)
b = np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])


a = np.zeros((2,2))
print(a)              # [[ 0.  0.]
                      # [ 0.  0.]]

b = np.ones((1,2))
print(b)              #[[ 1.  1.]]

c = np.full((2,2), 7)  # [[ 7.  7.]
print(c)               # [ 7.  7.]]

d = np.eye(2)         # [[ 1.  0.]
                      # [ 0.  1.]]

e = np.random.random((2,2))  #an array filled with random values
print(e)
#####################################################


################ Arrays Indexing ##################


#           Slicing: Similar to Python lists,     #
#           numpy arrays can be sliced.           #
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:, 1:3]
print(b)
print(a[0, 1])
b[0, 0] = 77
print(a[0, 1])



#           mix Slice and Integer Indexing       #
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

row_r1 = a[1, :] # rank = 1
row_r2 = a[1:2, :] # rank = 2
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)


#              Integer Array  Indexing            #
a = np.array([[1,2], [3, 4], [5, 6]])

print(a[[0, 1, 2], [0, 1, 0]])
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

print(a[[0, 0], [1, 1]])
print(np.array([a[0, 1], a[0, 1]]))
# Mutating one element from each row
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])
a[np.arange(4), b] += 10
print(a)



#             Boolean Array   Indexing           #
a = np.array([[-1,2], [3, 4], [-5, 6]])
bool_idx = (a < 0)
print(bool_idx)  # [[False False]
                 # [ True  True]
                 # [ True  True]]"

# rank = 1
print(a[bool_idx])
print(a > 2)
a[a < 0] = 0
print(a)


# https://numpy.org/doc/stable/reference/arrays.indexing.html #
###################################################


################## DataTypes ######################
#Note: all elements are the same type
x = np.array([1, 2])
print(x.dtype)         #int64

x = np.array([1.0, 2.0])
print(x.dtype)         #float64

x = np.array([1, 2], dtype=np.int64)
print(x.dtype)          #int64
####################################################




################# Array Math #######################
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise
#both produce the array
print(x + y)
print(np.add(x, y))

#both produce the array
print(x - y)
print(np.subtract(x, y))

#both produce the array
print(x * y)
print(np.multiply(x, y))

#both produce the array
print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))
# End Elementwise :)

# * -> Elementwise, dot -> matrix multiplication
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))

print(x.dot(v))
print(np.dot(x, v))

print(x.dot(y))
print(np.dot(x, y))

###### find the full list of mathematical functions here:   #
# https://numpy.org/doc/stable/reference/routines.math.html #
x = np.array([[1,2],[3,4]])

print(np.sum(x))  #10
print(np.sum(x, axis=0))  #[4 6]
print(np.sum(x, axis=1))  #[3 7]


#  more functions for manipulating arrays: #
# https://numpy.org/doc/stable/reference/routines.array-manipulation.html #
x = np.array([[1,2], [3,4]])
print(x)
print(x.T)  # [[1 3]
            # [2 4]]

# the transpose of a rank 1 array does nothing
v = np.array([1,2,3])
print(v)    # [1 2 3]
print(v.T)  # [1 2 3]
######################################################



################## BroadCasting  #####################
#ex1:
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
#x.shape = (4, 3), v.shape = (1, 3)
v = np.array([1, 0, 1])
y = np.empty_like(x)
for i in range(4):
    y[i, :] = x[i, :] + v
print(y)

#ex2:
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])

y = x + v
print(y)

#ex3:
w = np.array([4,5])

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])

print((x.T + w).T)
######################################################



## if you want to know more about numpy, see:        #
#  https://numpy.org/doc/stable/reference/           #
############# Image Operations (briefly) #############
from scipy.misc import imread, imsave, imresize

img = imread('assets/cat.jpg')
print(img.dtype, img.shape)
#img.shape -> (200, 200 , 3)  shape -> (1, 1, 3)
img_tinted = img * [1, 0.95, 0.9]
img_tinted = imresize(img_tinted, (300, 300))
imsave('assets/cat_tinted.jpg', img_tinted)
