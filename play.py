import numpy as np
import timeit

w = np.array([[0,0,1,0,0,1]
              ,[1,0,0,1,1,1]])
q = np.array([3, 2])
a = np.matrix([[0,0,1,0,0,1]
              ,[1,0,0,1,1,1]])
b = np.matrix([3, 2])

def dotProduct(a,b):

    return a * b[:, None]

# starttime = timeit.default_timer()
# print("The start time is :",starttime)
# dotProduct(w,q)
# print("The time difference is :", timeit.default_timer() - starttime)
starttime = timeit.default_timer()
print("The start time is :",starttime)
dotProduct(a,b)
print("The time difference is :", timeit.default_timer() - starttime)