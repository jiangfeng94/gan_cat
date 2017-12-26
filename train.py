# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:47:03 2017

@author: SalaFeng-
"""

import config
import glob
import os
import numpy as np
import scipy.misc
import tensorflow as tf
from skimage.io import imsave
import cv2
from PIL import Image

batch_size =config.batch_size
img_size =config.img_size
z_size =config.z_size
h1_size =config.h1_size
h2_size =config.h2_size
h3_size =config.h3_size


def show_result(batch_res, fname, grid_size=(4, 4), grid_pad=5):
    batch_res =  0.5*batch_res.reshape((batch_res.shape[0], 64, 64,3)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w,3), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res)*256
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w,:] = img
    imsave(fname, img_grid)


def build_generator(Z):
    w1 = tf.get_variable(name="g_w1", shape=[z_size, h1_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b1 = tf.get_variable(name="g_b1", shape=[h1_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(Z, w1) + b1)
    w2 = tf.get_variable(name="g_w2", shape=[h1_size, h2_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b2 = tf.get_variable(name="g_b2", shape=[h2_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    w3 = tf.get_variable(name="g_w3", shape=[h2_size, h3_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b3 = tf.get_variable(name="g_b3", shape=[h3_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

    w4 = tf.get_variable(name="g_w4", shape=[h3_size, img_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b4 = tf.get_variable(name="g_b4", shape=[img_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    h4 = tf.matmul(h3, w4) + b4
    x_generate = tf.nn.tanh(h4)
    g_params = [w1, b1, w2, b2, w3, b3, w4, b4]
    return x_generate, g_params


def build_discriminator(x_data, x_generator, keep_prob):
    x_in = tf.concat([x_data, x_generator], 0)
    w1 = tf.get_variable(name="d_w1", shape=[img_size, h3_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b1 = tf.get_variable(name="d_b1", shape=[h3_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)

    w2 = tf.get_variable(name="d_w2", shape=[h3_size, h2_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b2 = tf.get_variable(name="d_b2", shape=[h2_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

    w3 = tf.get_variable(name="d_w3", shape=[h2_size, h1_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b3 = tf.get_variable(name="d_b3", shape=[h1_size], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    h3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h2, w3) + b3), keep_prob)

    w4 = tf.get_variable(name="d_w4", shape=[h1_size, 1], initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=tf.float32)
    b4 = tf.get_variable(name="d_b4", shape=[1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    h4 = tf.matmul(h3, w4) + b4
    y_data = tf.nn.sigmoid(tf.slice(h4, [0, 0], [batch_size, -1], name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h4, [batch_size, 0], [-1, -1], name=None))
    d_params = [w1, b1, w2, b2, w3, b3, w4, b4]
    return y_data, y_generated, d_params


def get_image(image_path):
    image = Image.open(image_path) #.convert('L')  # 转化为灰度图
    arr_img =np.array(image)
    arr_img = arr_img.reshape([64 * 64 *3])
    return np.array(arr_img) / 256
    
def train():

    x_data =tf.placeholder(tf.float32,[batch_size,img_size],name ="x_data")
    Z =tf.placeholder(tf.float32,[batch_size,z_size],name ="Z")
    keep_prob =tf.placeholder(tf.float32,name ="keep_prob")
    
    x_generated,g_params =build_generator(Z)
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)
    
    d_loss = -tf.reduce_mean(tf.log(y_data) +tf.log(1-y_generated))
    g_loss = -tf.reduce_mean(tf.log(y_generated))
    
    optimizer=tf.train.AdamOptimizer(0.00005, beta1=0.9, beta2=0.999)
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)
    #初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    
    for epoch in range(config.epoch):
        data =glob.glob(os.path.join("./cats_64x64", "*.jpg"))
        batch_idxs =len(data) // batch_size
        for idx in range(batch_idxs):
            batch_files =data[idx*batch_size:(idx + 1) * batch_size]
            batch =[
                get_image(batch_file) for batch_file in batch_files]
            x_value = 2*np.array(batch).astype(np.float32)-1
            z_value = np.random.uniform(0, 1, size=(config.batch_size, config.z_size)).astype(np.float32)
            _,D_loss_curr=sess.run([d_trainer,d_loss],feed_dict ={x_data:x_value,Z:z_value,keep_prob: np.sum(0.7).astype(np.float32)})
            _,G_loss_curr=sess.run([g_trainer,g_loss],feed_dict={x_data: x_value,Z:z_value,keep_prob: np.sum(0.7).astype(np.float32)})
            print('Epoch :{}  D_loss:{:0.4f} G_loss:{:0.4f}'.format(epoch,D_loss_curr,G_loss_curr))
        x_gen_val = sess.run(x_generated, feed_dict={Z: z_sample_val})
        path ="output"
        if not os.path.exists(path):
            os.mkdir(path)
            print("makedir----->{}".format(path))
        show_result(x_gen_val, "output/10_20_40_60_xavier/epoch{}.jpg".format(epoch))
            




train()