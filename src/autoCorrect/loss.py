from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from keras import backend as K


def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)


def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)


class NB(object):
    def __init__(self, theta=None, theta_default=None, theta_init=[0.0],
                 masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False, out_idx=None):

        # for numerical stability
        self.eps = 1e-10
        self.lambd = 1e-3
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta
        self.theta_default = theta_default
        self.out_idx = out_idx

        with tf.name_scope(self.scope):
            # a variable may be given by user or it can be created here
            if theta is None:
                theta = tf.Variable(theta_init, dtype=tf.float32,
                                    name='theta')
            if theta_default is None:
                self.theta_default = tf.constant([25.0])

            # keep a reference to the variable itself
            self.theta_variable = theta

            # to keep dispersion always non-negative
            self.theta = tf.nn.softplus(theta)
            #self.theta = theta
            if self.out_idx is not None:
                self.out_idx = tf.cast(self.out_idx, tf.bool)

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        #eps = self.eps
        lambd = self.lambd
        power = tf.constant([2.0])

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            theta = self.theta    
            # Clip theta
            #theta = tf.maximum(self.theta, 1e-6)
            #theta = tf.minimum(self.theta, 1e6) #
            #theta = 1.0/(self.theta+eps)
            #theta = self.theta+eps

            t1 = -tf.lgamma(y_true+theta)
            t2 = tf.lgamma(theta)
            t3 = tf.lgamma(y_true+1.0)
            t4 = -(theta * (tf.log(theta)))
            t5 = -(y_true * (tf.log(y_pred)))
            t6 = (theta+y_true) * tf.log(theta+y_pred)
            #t7 = lambd*tf.pow((theta-self.theta_default),power)

            assert_ops = [
                    tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                    tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                    tf.verify_tensor_all_finite(t2, 't2 has inf/nans'),
                    tf.verify_tensor_all_finite(t3, 't3 has inf/nans'),
                    tf.verify_tensor_all_finite(t4, 't4 has inf/nans'),
                    tf.verify_tensor_all_finite(t5, 't5 has inf/nans'),
                    tf.verify_tensor_all_finite(t6, 't6 has inf/nans'),
                    #tf.verify_tensor_all_finite(t7, 't7 has inf/nans')
            ]

            if self.debug:
                tf.summary.histogram('t1', t1)
                tf.summary.histogram('t2', t2)
                tf.summary.histogram('t3', t3)
                tf.summary.histogram('t4', t4)
                tf.summary.histogram('t5', t5)
                tf.summary.histogram('t6', t6)
                #tf.summary.histogram('t7', t7)

                with tf.control_dependencies(assert_ops):
                    final = t1 + t2 + t3 + t4 + t5 + t6 #+ t7

            else:
                final = t1 + t2 + t3 + t4 + t5 + t6 #+ t7

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                     if self.out_idx is not None:
                        final = tf.boolean_mask(final, self.out_idx)    
                        final = tf.reduce_mean(final)
            final = _nan2inf(final)            
        return final



#def mean_squared_error_exp(y_true, y_pred):
#    return K.mean(K.square(y_pred - np.log1p(y_true)), axis=-1)
