#adapted from https://gist.github.com/myme5261314/005ceac0483fc5a581cc

import tensorflow as tf
import numpy as np
import input_data
from PIL import Image
from Util import tile_raster_images

alpha = 0.5
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

rbm_w = tf.placeholder("float", [784, 1000])
rbm_vb = tf.placeholder("float", [784])
rbm_hb = tf.placeholder("float", [1000])
h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v1 = sample_prob(tf.nn.sigmoid(
    tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
update_w = rbm_w + alpha * \
    (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

#RBM - bi partite
def RBM_2( hid,vis):
    hidden = len(hid)    #number of hidden units
    visible = len(vis)  #number of visible

    #initialize weights and biases
    weight =  np.random.rand(visible,hidden)
    bias_v = np.random.rand(visible,1)
    bias_h = np.random.rand(visible,1)

    # define Energy

    energy = - np.dot(bias_v,vis)-np.dot(bias_h,hid)- np.dot(vis,np.dot(weight,hid))

    if start % 10000 == 0:
        print(sess.run(
            err_sum, feed_dict={X: trX, rbm_w: n_w, rbm_vb: n_vb, rbm_hb: n_hb}))
        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save("rbm_%d.png" % (start / 100000))    
