# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:37:03 2017

@author: SalaFeng-
"""

import config
import glob
import os
import numpy as np
import tensorflow as tf
import cv2
from skimage.io import imsave
from PIL import Image

batch_size =config.batch_size
img_size =config.img_size
z_size =config.z_size
def get_image(image_path):
    image = Image.open(image_path) #.convert('L')  # 转化为灰度图
    arr_img =np.array(image)
    arr_img = arr_img.reshape([64 * 64 *3])
    return np.array(arr_img) / 256
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




def leakyrelu(x, leak=0.2, name="leakyrelu"):
    return tf.maximum(x, leak * x)

def conv2d(input,output_dim,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="conv2d"):
    w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv =tf.nn.conv2d(input,w,strides=[1,d_h,d_w,1],padding="SAME")
    bias =tf.get_variable('b',[output_dim],initializer=tf.constant_initializer(0.0))
    conv =tf.reshape(tf.nn.bias_add(conv,bias),conv.get_shape())
    return conv

def build_discriminator(x_data,x_generated):
    w0    = tf.get_variable('w0',[5,5,3,64],initializer=tf.truncated_normal_initializer(stddev=0.02))  # 3 ==x_data[-1]
    conv0 = tf.nn.conv2d(x_data,w0,strides=[1,2,2,1],padding="SAME")
    b0    = tf.get_variable('b0',[64],initializer=tf.constant_initializer(0.0))
    conv0 = tf.reshape(tf.nn.bias_add(conv0,b0),conv0.get_shape())
    h0    = leakyrelu(conv0)

    w1    = tf.get_variable('w1',[5,5,3,128],initializer=tf.truncated_normal_initializer(stddev=0.02))  # 3 ==x_data[-1]
    conv1 = tf.nn.conv2d(x_data,w1,strides=[1,2,2,1],padding="SAME")
    b1    = tf.get_variable('b1',[128],initializer=tf.constant_initializer(0.0))
    conv1 = tf.reshape(tf.nn.bias_add(conv1,b1),conv1.get_shape())
    h1    = leakyrelu(conv1)


def build_generator(Z):
    ghu=1
def train():
    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")

    Z = tf.placeholder(tf.float32, [batch_size, z_size], name="Z")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    x_generated, g_params = build_generator(Z)
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

    d_loss = -tf.reduce_mean(tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = -tf.reduce_mean(tf.log(y_generated))

    optimizer = tf.train.AdamOptimizer(0.00005, beta1=0.9, beta2=0.999)
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)
    # 初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    for epoch in range(config.epoch):
        data = glob.glob(os.path.join("./cats_64x64", "*.jpg"))
        batch_idxs = len(data) // batch_size
        for idx in range(batch_idxs):
            batch_files = data[idx * batch_size:(idx + 1) * batch_size]
            batch = [
                get_image(batch_file) for batch_file in batch_files]
            x_value = 2 * np.array(batch).astype(np.float32) - 1
            z_value = np.random.uniform(0, 1, size=(config.batch_size, config.z_size)).astype(np.float32)
            _, D_loss_curr = sess.run([d_trainer, d_loss], feed_dict={x_data: x_value, Z: z_value,
                                                                      keep_prob: np.sum(0.7).astype(np.float32)})
            _, G_loss_curr = sess.run([g_trainer, g_loss], feed_dict={x_data: x_value, Z: z_value,
                                                                      keep_prob: np.sum(0.7).astype(np.float32)})
           # print('Epoch :{}  D_loss:{:0.4f} G_loss:{:0.4f}'.format(epoch, D_loss_curr, G_loss_curr))
        x_gen_val = sess.run(x_generated, feed_dict={Z: z_sample_val})
        path = "output"
        if not os.path.exists(path):
            os.mkdir(path)
            print("makedir----->{}".format(path))
        show_result(x_gen_val, "output/10_20_40_60_xavier/epoch{}.jpg".format(epoch))
train()