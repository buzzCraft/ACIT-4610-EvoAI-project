import numpy as np
import timeit
from matplotlib import pyplot as plt
#
# w = np.array([[0,0,1,0,0,1]
#               ,[1,0,0,1,1,1]])
# q = np.array([3, 2])
#
#
# def dotProduct(a,b):
#
#     return a * b[:, None]
#
# # starttime = timeit.default_timer()
# # print("The start time is :",starttime)
# # dotProduct(w,q)
# # print("The time difference is :", timeit.default_timer() - starttime)
# # starttime = timeit.default_timer()
# # print("The start time is :",starttime)
# # dotProduct(a,b)
# # print("The time difference is :", timeit.default_timer() - starttime)
# #
# # input_train = w * q[:, None]
# # print(input_train)
# # input = np.sum(input_train, axis=0)
# # print("Nr 2")
# # print(input)
# # # n.update_list(input)
# #
# #
# starttime = timeit.default_timer()
# print("The start time is :",starttime)
# x = np.array([])
# for i in range(100):
#     x = np.append(x, i)
# print(x)
# print("The time difference is :", timeit.default_timer() - starttime)
# starttime = timeit.default_timer()
# print("The start time is :",starttime)
# y = np.zeros(100)  # Raskest!!!!
# for i in range(100):
#     y[i] = i
# print(y)
# print("The time difference is :", timeit.default_timer() - starttime)
# t = 100
#
# n = 784
#
# a = np.zeros((n,t))
# z = np.array([1,2,3,4,5])
# a[0] = z
# print(a)

# print(np.ones(10))
# import random
# liste = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# random.shuffle(liste)
# def getSublists(lst,n):
#     subListLength = len(lst) // n
#     for i in range(0, len(lst), subListLength):
#         yield lst[i:i+subListLength]
#
# x = list(getSublists(liste, 4))
# print(x)
# # print(list(getSublists(liste,3)))
#
#
# x = [1,2,3]
#
# for i in range(10):
#     if i not in x:
#         print(i)
import time

x = np.float(3.1)
u = 3.1
y = 2.3
z = np.float(2.3)

start = time.time()
for i in range(1000000):
    l = u * y
print(l)
end = time.time()
delta = end - start

print("Time: ", delta)

start = time.time()
for i in range(1000000):
    l = x * z
print(l)
end = time.time()
delta = end - start
print("Time: ", delta)
