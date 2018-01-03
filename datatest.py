# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:46:15 2018

@author: SalaFeng-
"""
import numpy as np 
import tensorflow as tf
'''
mat=[[0 for i in range(12288)] for j in range(64)]
arr1 =np.array(mat)
mat2 =[[0 for i in range(64*8*4*4)] for j in range(64*64*3)]
arr2 =np.array(mat2)
print(arr1.shape)
print(arr2.shape)
'''
a =np.ones([16,64*64*3])
b =np.ones([64*64*3,64*8*4*4])
c =np.ones(64*8*4*4)
print(a.shape)
print(b.shape)

z_ =tf.matmul(a,b)+c
print(z_.shape)

