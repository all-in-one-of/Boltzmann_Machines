#adapted from https://gist.github.com/myme5261314/005ceac0483fc5a581cc

import tensorflow as tf
import numpy as np
import input_data
from PIL import Image
from Util import tile_raster_images
import Simulated_Annealing as sim

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


#split sampling of hiddenlayer into two groups
#train as normal RBM ignoring tripartite connections. Then use this to bring whole system into thermal equilibrium. Update all weights

#train RBM

#Gibbs sampling for standard RBM
h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v1 = sample_prob(tf.nn.sigmoid(
    tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)
#weight update
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
update_w = rbm_w + alpha * \
    (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

#calculating error
h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v_sample = sample_prob(tf.nn.sigmoid(
    tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb))
err = X - v_sample
err_sum = tf.reduce_mean(err * err)

#train entire system with simulated annealing
#energy functional of complete system
h01=tf.split(h0,2,1)[0]
h02=tf.split(h0,2,1)[1]

energy = tf.matmul(X,tf.transpose(tf.expand_dims(rbm_vb,0)))+tf.matmul(h0,tf.transpose(tf.expand_dims(rbm_hb,0)))\
         +tf.matmul(tf.matmul((X),rbm_w),tf.transpose(h0))+tf.matmul(tf.matmul(tf.transpose(h01),rbm_w3),h02)

#clamp visible layer
h_eq1 = sim.sim_an(h0,100,energy)

#free run on all layers
v_eq,h_eq2 = sim.sim_an([X,h_eq1],100,energy)

#weightupdate RBM(simplified, potentially erratic learning)
w_positive_grad_3 = tf.matmul(tf.transpose(X,h_eq1))
w_negative_grad_3 = tf.matmul(tf.transpose(v_eq,h_eq2))
update_w_3=rbm_w+alpha*(w_positive_grad_3-w_negative_grad_3)
update_vb_3 = rbm_vb +alpha*tf.reduce_mean(X-v_eq,0)
update_hb_3 = rbm_hb+alpha*tf.reduce_mean(h_eq1-h_eq2,0)
#weightupdate of tripartite connections


#initializing session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

n_w = np.zeros([784, 500], np.float32)
n_vb = np.zeros([784], np.float32)
n_hb = np.zeros([500], np.float32)
n_w3 = np.zeros([250,250],np.float32)
o_w = np.zeros([784, 500], np.float32)
o_vb = np.zeros([784], np.float32)
o_hb = np.zeros([500], np.float32)
o_w3 = np.zeros([250,250], np.float32)
print(sess.run(err_sum, feed_dict={X: trX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb,rbm_w3:o_w3}))

#run training and use generativity (RBM)
for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
    batch = trX[start:end]
    n_w = sess.run(update_w, feed_dict={
                   X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb, rbm_w3:o_w3})
    n_vb = sess.run(update_vb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb,rbm_w3:o_w3})
    n_hb = sess.run(update_hb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb,rbm_w3:o_w3})
    #n_w3 = sess.run(update_w3, feed_dict={X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb,rbm_w3:o_w3})
    o_w = n_w
    o_vb = n_vb
    o_hb = n_hb
    #o_w3 = n_w3
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

#0.0836942
#train entire system with weights initialized as result from rbm training
