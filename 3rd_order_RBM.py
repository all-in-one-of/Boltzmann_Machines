#adapted from https://gist.github.com/myme5261314/005ceac0483fc5a581cc

import tensorflow as tf
import numpy as np
import input_data
from PIL import Image
from Util import tile_raster_images

def sample_prob(probs):
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(tf.shape(probs))))

alpha = 0.5
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#input and output
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])


#weights and biases of standard RBM
rbm_w = tf.placeholder("float", [784, 500])
rbm_vb = tf.placeholder("float", [784])
rbm_hb = tf.placeholder("float", [500])

#weights of additional 3-partite connection
rbm_w3 = tf.placeholder("float",[250,250])


#Gibbs sampling for standard RBM
#split sampling of hiddenlayer into two groups 
ha0 = sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[0]+tf.matmul(sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[1])),rbm_w3)+tf.split(rbm_hb,2)[0]))
hb0 = sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[1]+tf.matmul(sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[0])),rbm_w3)+tf.split(rbm_hb,2)[1]))
h0 = tf.concat([ha0,hb0],1)
v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))
ha1 = sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[0]+tf.matmul(sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[1])),rbm_w3)+tf.split(rbm_hb,2)[0]))
hb1 = sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[1]+tf.matmul(sample_prob(tf.nn.sigmoid(tf.split(tf.matmul(X,rbm_w),2,1)[0])),rbm_w3)+tf.split(rbm_hb,2)[1]))
h1 = tf.concat([ha1,hb1],1)



#updating rules of standard RBM
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
update_w = rbm_w + alpha * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

#claculating error
#sample
h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb))
#error
err = X - v_sample
err_sum = tf.reduce_mean(err * err)

#initializing session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

n_w = np.zeros([784, 500], np.float32)
n_vb = np.zeros([784], np.float32)
n_hb = np.zeros([500], np.float32)
o_w = np.zeros([784, 500], np.float32)
o_vb = np.zeros([784], np.float32)
o_hb = np.zeros([500], np.float32)
o_w3 = np.zeros([250,250], np.float32)
print(sess.run(err_sum, feed_dict={X: trX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb,rbm_w3:o_w3}))

#run training and use generativity
for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
    batch = trX[start:end]
    n_w = sess.run(update_w, feed_dict={
                   X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb, rbm_w3:o_w3})
    n_vb = sess.run(update_vb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb,rbm_w3:o_w3})
    n_hb = sess.run(update_hb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb,rbm_w3:o_w3})
    o_w = n_w
    o_vb = n_vb
    o_hb = n_hb
    if start % 10000 == 0:
        print(sess.run(err_sum, feed_dict={X: trX, rbm_w: n_w, rbm_vb: n_vb, rbm_hb: n_hb, rbm_w3:o_w3}))
        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.show()
        
