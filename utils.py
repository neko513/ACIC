""" Utility functions. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import random
import re
from tensorflow.contrib.layers.python import layers as tf_layers
FLAGS = tf.app.flags.FLAGS

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    # Creates an optimizer training op.
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        
    
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op

class AdamWeightDecayOptimizer(tf.train.Optimizer):
    # A basic Adam optimizer that includes "correct" L2 weight decay.
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
    # Constructs a AdamWeightDecayOptimizer.
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        # See base class.
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                m.assign(next_m),
                v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        # Whether to use L2 weight decay for `param_name`.
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        # Get the variable name from the tensor name.
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride = [1,2,2,1]
    conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
   
    return normed

def normalize(inp, activation, reuse, scope):
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    
## Loss functions
def PW_loss(pred, label,threshold=0, Tag=0):   
    pred = tf.reshape(pred, [-1])
    label = label[0:Tag,0:Tag]
    label = tf.reshape(label, [-1])
    weight = tf.square(2 * label - tf.ones(tf.shape(label)))
    square = tf.square(pred - label)
    return tf.reduce_mean(tf.multiply(weight, square))  # /FLAGS.batch







