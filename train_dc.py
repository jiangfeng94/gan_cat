# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:37:03 2017

@author: SalaFeng-
"""

import myconfig
import glob
import os
import numpy as np
import tensorflow as tf

from skimage.io import imsave
from PIL import Image

batch_size=myconfig.batch_size
img_size =myconfig.img_size
z_size =myconfig.z_size
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
def build_discriminator(x_data, x_generated):
    x_data=tf.reshape(x_data,[batch_size, 64, 64, 3])
    x_in = tf.concat([x_data, x_generated], 0)
    d_w0 = tf.get_variable('d_w0', [5, 5, x_in.get_shape()[-1], 64],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_conv0 = tf.nn.conv2d(x_in, d_w0, strides=[1, 2, 2, 1], padding='SAME')
    d_b0 = tf.get_variable('d_b0', [64], initializer=tf.constant_initializer(0.0))
    h0 = tf.reshape(tf.nn.bias_add(d_conv0, d_b0), d_conv0.get_shape())
    h0 = leakyrelu(h0)

    d_w1 = tf.get_variable('d_w1', [5, 5, h0.get_shape()[-1], 128],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_conv1 = tf.nn.conv2d(h0, d_w1, strides=[1, 2, 2, 1], padding='SAME')
    d_b1 = tf.get_variable('d_b1', [128], initializer=tf.constant_initializer(0.0))
    h1 = tf.reshape(tf.nn.bias_add(d_conv1, d_b1), d_conv1.get_shape())
    bn1 = tf.contrib.layers.batch_norm(h1)
    h1 = leakyrelu(bn1)

    d_w2 = tf.get_variable('d_w2', [5, 5, h1.get_shape()[-1], 256],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_conv2 = tf.nn.conv2d(h1, d_w2, strides=[1, 2, 2, 1], padding='SAME')
    d_b2 = tf.get_variable('d_b2', [256], initializer=tf.constant_initializer(0.0))
    h2 = tf.reshape(tf.nn.bias_add(d_conv2, d_b2), d_conv2.get_shape())
    bn2 = tf.contrib.layers.batch_norm(h2)
    h2 = leakyrelu(bn2)

    d_w3 = tf.get_variable('d_w3', [5, 5, h2.get_shape()[-1], 512],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_conv3 = tf.nn.conv2d(h2, d_w3, strides=[1, 2, 2, 1], padding='SAME')
    d_b3 = tf.get_variable('d_b3', [512], initializer=tf.constant_initializer(0.0))
    h3 = tf.reshape(tf.nn.bias_add(d_conv3, d_b3), d_conv3.get_shape())
    bn3 = tf.contrib.layers.batch_norm(h3)
    h3 = leakyrelu(bn3)

    h3 =tf.reshape(h3,[batch_size,-1])
    shape = h3.get_shape().as_list()

    d_w4 = tf.get_variable("d_w4", [shape[1], 1], tf.float32, tf.random_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable("d_b4", [1], initializer=tf.constant_initializer(0.0))
    h4 = tf.matmul(h3, d_w4) + d_b4

    y_data = tf.nn.sigmoid(tf.slice(h4, [0, 0], [batch_size, -1], name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h4, [batch_size, 0], [-1, -1], name=None))

    d_params = [d_w0, d_b0,d_w1, d_b1,d_w2, d_b2,d_w3, d_b3,d_w4, d_b4]
    return y_data, y_generated, d_params
                         

def build_generator(Z):
    g_w0 =tf.get_variable("g_w0",[z_size,8192],tf.float32,tf.random_normal_initializer(stddev=0.02))
    g_b0 =tf.get_variable("g_b0", [8192],initializer=tf.constant_initializer(0.0))
    #Z=[16,100]  g_w0 =[100,8192]
    z_ =tf.matmul(Z, g_w0) + g_b0  #shape =[64,8192]
    h0 =tf.reshape(z_,[-1,4,4,512])   #h0 =[64,4,4,512]
    bn0 =tf.contrib.layers.batch_norm(h0)
    h0 =tf.nn.relu(bn0)
    # 第二个参数依次为卷积核的高，宽，输出的特征图个数，输入的特征图个数。
    g_w1 =tf.get_variable("g_w1",[5,5,256,512],initializer=tf.random_normal_initializer(stddev=0.02))
    deconv = tf.nn.conv2d_transpose(h0, g_w1, output_shape=[batch_size,8,8,256],strides=[1, 2, 2, 1])
    g_b1 = tf.get_variable('g_b1', [256], initializer=tf.constant_initializer(0.0))
    h1 = tf.reshape(tf.nn.bias_add(deconv, g_b1), deconv.get_shape())
    bn1 = tf.contrib.layers.batch_norm(h1)
    h1 = tf.nn.relu(bn1)

    g_w2 =tf.get_variable("g_w2",[5,5,128,256],initializer=tf.random_normal_initializer(stddev=0.02))
    deconv = tf.nn.conv2d_transpose(h1, g_w2, output_shape=[batch_size,16,16,128],strides=[1, 2, 2, 1])
    g_b2 = tf.get_variable('g_b2', [128], initializer=tf.constant_initializer(0.0))
    h2 = tf.reshape(tf.nn.bias_add(deconv, g_b2), deconv.get_shape())
    bn2 = tf.contrib.layers.batch_norm(h2)
    h2 = tf.nn.relu(bn2)

    g_w3 =tf.get_variable("g_w3",[5,5,64,128],initializer=tf.random_normal_initializer(stddev=0.02))
    deconv = tf.nn.conv2d_transpose(h2, g_w3, output_shape=[batch_size,32,32,64],strides=[1, 2, 2, 1])
    g_b3 = tf.get_variable('g_b3', [64], initializer=tf.constant_initializer(0.0))
    h3 = tf.reshape(tf.nn.bias_add(deconv, g_b3), deconv.get_shape())
    bn3 = tf.contrib.layers.batch_norm(h3)
    h3 = tf.nn.relu(bn3)

    g_w4 = tf.get_variable("g_w4", [5, 5, 3, 64], initializer=tf.random_normal_initializer(stddev=0.02))
    deconv = tf.nn.conv2d_transpose(h3, g_w4, output_shape=[batch_size, 64, 64, 3], strides=[1, 2, 2, 1])
    g_b4 = tf.get_variable('g_b4', [3], initializer=tf.constant_initializer(0.0))
    h4 = tf.reshape(tf.nn.bias_add(deconv, g_b4), [batch_size, 64, 64, 3])

    g_params = [g_w0, g_b0,g_w1, g_b1,g_w2, g_b2,g_w3, g_b3,g_w4, g_b4]
    return tf.nn.tanh(h4),g_params
    
    
def train():
    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")

    Z = tf.placeholder(tf.float32, [batch_size, z_size], name="Z")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    x_generated,g_params = build_generator(Z)
    y_data, y_generated,d_params= build_discriminator(x_data, x_generated)

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

    for epoch in range(myconfig.epoch):
        data = glob.glob(os.path.join("./cats_64x64", "*.jpg"))
        batch_idxs = len(data) // batch_size
        for idx in range(batch_idxs):
            batch_files = data[idx * batch_size:(idx + 1) * batch_size]
            batch = [
                get_image(batch_file) for batch_file in batch_files]
            x_value = 2 * np.array(batch).astype(np.float32) - 1
            z_value = np.random.uniform(0, 1, size=(myconfig.batch_size, myconfig.z_size)).astype(np.float32)
            print(z_value.shape)
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