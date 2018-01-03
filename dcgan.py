import tensorflow as tf
import tensorflow.contrib.layers as tcl
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import scipy.misc
import matplotlib.gridspec as gridspec
def lrelu(x, alpha=0.2):
	return tf.maximum(tf.minimum(0.0, alpha * x), x
)
def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

def get_img(img_path, crop_h, resize_h):
	img=scipy.misc.imread(img_path).astype(np.float)
	# crop resize
	crop_w = crop_h
	#resize_h = 64
	resize_w = resize_h
	h, w = img.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
	return np.array(cropped_image)/255.0

class getdata():
    def __init__(self):
        datapath = '/cats_64x64'
        self.z_dim = 100
        self.c_dim = 2
        self.size = 64
        self.channel = 3
        self.data = glob(os.path.join(datapath, '*.jpg'))
        self.batch_count = 0
    def __call__(self, batch_size):
        batch_number = len(self.data) / batch_size
        if self.batch_count < batch_number - 1:
            self.batch_count += 1
        else:
            self.batch_count = 0
        path_list = self.data[self.batch_count * batch_size:(self.batch_count + 1) * batch_size]
        batch = [get_img(img_path, 256, self.size) for img_path in path_list]
        batch_imgs = np.array(batch).astype(np.float32)

        return batch_imgs

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)
        return fig

class G_conv(object):
    def __init__(self):
        self.name ='G_conv'
        self.size =4
        self.channel =3
    def __call__(self,z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, self.size * self.size * 1024, activation_fn=tf.nn.relu,
                                    normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, (-1, self.size, self.size, 1024))  # size
            g = tcl.conv2d_transpose(g, 512, 3, stride=2,  # size*2
                                     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 256, 3, stride=2,  # size*4
                                     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 128, 3, stride=2,  # size*8
                                     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, self.channel, 3, stride=2,  # size*16
                                     activation_fn=tf.nn.sigmoid, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
class D_conv(object):
    def __init__(self):
        self.name = 'D_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4,  # bzx64x64x3 -> bzx32x32x64
                                stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4,  # 16x16x128
                                stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4,  # 8x8x256
                                stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4,  # 4x4x512
                                stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.flatten(shared)

            d = tcl.fully_connected(shared, 1, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 2, activation_fn=None)  # 10 classes
            return d, q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class DCGAN():
    def __init__(self,generator,discriminator,data):
        self.generator =generator
        self.discriminator = discriminator
        self.data =data
        #data
        self.z_dim =self.data.z_dim
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        #nets
        self.G_sample =self.generator(self.z)
        self.D_real,_ =self.discriminator(self.X)
        self.D_fake,_ =self.discriminator(self.G_sample,reuse=True)

        #loss
        self.D_loss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                    (logits=self.D_real,labels =tf.ones_like(self.D_real)))
        self.G_loss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                    (logits=self.D_fake,labels =tf.ones_like(self.D_real)))
        #solver
        print(discriminator.vars)
        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4).\
            minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4)\
            .minimize(self.G_loss, var_list=self.generator.vars)

        self.saver =tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess =tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    def train(self,output_dir,ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 32):
        count =0
        self.sess.run(tf.global_variables_initializer())
        for epoch in range (training_epoches):
            X_b =self.data(batch_size)
            self.sess.run(self.D_solver,
                          feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
            k = 1
            for _ in range(k):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.z: sample_z(batch_size, self.z_dim)}
                )
                # save img, model. print loss
                if epoch % 100 == 0 or epoch < 100:
                    D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
                    G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.z: sample_z(batch_size, self.z_dim)})
                    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

                    if epoch % 1000 == 0:
                        samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})

                        fig = self.data.data2fig(samples)
                        plt.savefig('{}/{}.png'.format(output_dir, str(count).zfill(3)), bbox_inches='tight')
                        count += 1
                        plt.close(fig)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    generator =G_conv()
    discriminator =D_conv()
    data = getdata()
    dcgan =DCGAN(generator,discriminator,data)

    output_dir ='output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dcgan.train(output_dir)