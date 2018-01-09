#adapted from https://gist.github.com/myme5261314/005ceac0483fc5a581cc

import tensorflow as tf
import numpy as np
import input_data
from PIL import Image
from Util import tile_raster_images



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
