"""
NN Policy with KL Divergence Constraint

                                   ->PolicyNN
Overview of functions: Policy->TRPO->LogProb
                                   ->KLEntropy

Written by Patrick Coady (pat-coady.github.io)
"""
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.optimizers import Adam
import numpy as np

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True) # solves the 'tried to create variables on non-first call' issue

# from keras.utils.vis_utils import plot_model
from fractalnet_regularNN import *


from tensorflow.python.ops import summary_ops_v2


# Called in train.py in line 292. This is the highest level function
class Policy(object):
    def __init__(self, bl,c,layersizes,dropout,deepest,obs_dim,kl_targ, init_logvar):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim (default is 10)
            init_logvar: natural log of initial policy variance
        """
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.epochs = 20
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.trpo = TRPO(bl,c,layersizes,dropout,deepest,obs_dim, kl_targ, init_logvar, eta)
        self.policy = self.trpo.get_layer('policy_nn') # output layer which gives the mean. (var is also given by this but still unsure how as it is not given by the NN, but is computed by the KERAS package somehow)
        # plot_model(self.trpo,to_file='model.png',show_shapes=True)
        self.lr = self.policy.get_lr()  # lr calculated based on size of PolicyNN
        self.trpo.compile(optimizer=Adam(self.lr * self.lr_multiplier)) # Configures model for training
        
        logdir = "/subclassed_model_logdir"
        xs = np.ones([29,])
        ys = np.zeros([8,])
        self.trpo.fit(xs,ys,epochs=1,callbacks=tf.keras.callbacks.TensorBoard(logdir)) 
        writer = summary_ops_v2.create_file_writer_v2(logdir)
        with writer.as_default():
            summary_ops_v2.graph(self.trpo.call.get_concrete_function(xs).graph)  # <--
            writer.flush()

        self.logprob_calc = LogProb()
        # plot_model(self.model, to_file='model.png', show_shapes=True)


        # self.replay_buffer_x = None
        # self.replay_buffer_y = None

    def sample(self, obs):
        """Draw sample from policy."""
        act_means, act_logvars = self.policy(obs) # give the current state (obs) and it will output what the current estimate for the mean and var is for that state
        act_stddevs = np.exp(act_logvars / 2)
        temp = np.random.normal(act_means, act_stddevs).astype(np.float32)
        return temp
    
    def update(self, observes, actions, advantages, logger, disc_sum_rew):
        # x = observes
        # y = disc_sum_rew
        # tf.TensorArray(x.dtype, 0, dynamic_size=True)
        # tf.executing_eagerly()
        # print('tf.executing_eagerly')
        # print(tf.executing_eagerly)
        # input('')
        # num_batches = max(x.shape[0] // 256, 1)
        # print('num_batches')
        # print(num_batches)
        # batch_size = x.shape[0] // num_batches
        # print('batch_size')
        # print(batch_size)
        # """ Update policy based on observations, actions and advantages

        # Args:
        #     observes: observations, shape = (N, obs_dim)
        #     actions: actions, shape = (N, act_dim)
        #     advantages: advantages, shape = (N,)
        #     logger: Logger object, see utils.py
        # """
        # if self.replay_buffer_x is None:
        #     x_train, y_train = x, y
        # else:
        #     x_train = np.concatenate([x, self.replay_buffer_x])
        #     y_train = np.concatenate([y, self.replay_buffer_y])
        # self.replay_buffer_x = x
        # self.replay_buffer_y = y

        # self.trpo.fit(x_train, y_train, epochs=self.epochs, batch_size=batch_size,
        #                shuffle=True, verbose=0, callbacks=[self.tbCallBack])
        # self.trpo.summary()
        # print('Model')
        # print(self.trpo)
        # input('')


        K.set_value(self.trpo.optimizer.lr, self.lr * self.lr_multiplier)
        K.set_value(self.trpo.beta, self.beta)
        old_means, old_logvars = self.policy(observes)
        old_means = old_means.numpy()
        old_logvars = old_logvars.numpy()
        old_logp = self.logprob_calc([actions, old_means, old_logvars])
        old_logp = old_logp.numpy()
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # 'train_on_batch' and 'predict_on_batch' are both functions
            # within Keras which was called with 'Model' input to 'TRPO'
            loss = self.trpo.train_on_batch([observes, actions, advantages,
                                             old_means, old_logvars, old_logp])
            kl, entropy = self.trpo.predict_on_batch([observes, actions, advantages,
                                                      old_means, old_logvars, old_logp])
            kl, entropy = np.mean(kl), np.mean(entropy)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})

class PolicyNN(Layer):
    """ Neural net for policy approximation function.

    Policy parameterized by Gaussian means and variances. NN outputs mean
     action based on observation. Trainable variables hold log-variances
     for each action dimension (i.e. variances not determined by NN).
     Variances can be calculated just with the vector containing all of 
     the probabilities and the mean from the NN.
    """
    def __init__(self, bl,c,layersizes, dropout,deepest, obs_dim, init_logvar, **kwargs):
        super(PolicyNN, self).__init__(**kwargs)
        self.bl = bl
        self.c = c
        self.layersizes = layersizes
        self.dropout = dropout
        self.deepest = deepest
        self.batch_sz = None
        self.init_logvar = init_logvar
        hid1_units = 512
        hid3_units = 32  # 10 empirically determined
        hid2_units = 128
        self.lr = 9e-4 / np.sqrt(hid2_units)  # 9e-4 empirically determined
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')

        # 'Dense' functions within Keras using tanh activation to compute
        # ouput = activation(dot(input,kernel) + bias) where kernel is a
        # weights matrix created by the layer
        
        self.dense1 = Dense(512, activation='relu', input_shape=(29,))
        self.dense2 = Dense(128, activation='relu', input_shape=(512,)) # hid1_units = 270 for halfcheetah
        self.dense3 = Dense(32, activation='relu', input_shape=(128,)) # hid2_units = 127 for halfcheetah
        self.dense4 = Dense(8, input_shape=(32,)) # hid3_units = 60 for halfcheetah
        # logvar_speed increases learning rate for log-variances.
        # heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_units) // 48
        logvar_speed = 16
        # 'add_weight' creates a trainable weight variable for this layer
        self.logvars = self.add_weight(shape=(logvar_speed, int(self.layersizes[4][0])),
                                       trainable=True, initializer='zeros')
        print('Policy Params -- lr: {:.3g}, logvar_speed: {}'
              .format(self.lr, logvar_speed))

    def build(self, input_shape):
        self.batch_sz = input_shape[0] # input_shape = (1,27)

    def call(self, inputs, **kwargs):
        # print("inputs") # inputs, obs_dim (1,27) tensor for halfcheetah
        # print(inputs)
        # y = self.dense1(inputs) # hid1 (1,270) tensor for halfcheetah
        # y = self.dense2(y) # hid2 (1,127) tensor for halfcheetah
        # y = self.dense3(y) # hid3 (1,60) tensor for halfcheetah
        # means = self.dense4(y) # (1,6) tensor for halfcheetah. 
        means = fractal_net(self,bl=self.bl,c=self.c,layersizes=self.layersizes,
            drop_path=0.15,dropout=self.dropout,
            deepest=self.deepest)(inputs)
        logvars = K.sum(self.logvars, axis=0, keepdims=True) + self.init_logvar
        logvars = K.tile(logvars, (self.batch_sz, 1))
       
        return [means, logvars]

    def get_lr(self):
        return self.lr


class KLEntropy(Layer):
    """
    Layer calculates:
        1. KL divergence between old and new policy distributions
        2. Entropy of present policy

    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
    """
    def __init__(self, **kwargs):
        super(KLEntropy, self).__init__(**kwargs)
        self.act_dim = None

    def build(self, input_shape):
        self.act_dim = input_shape[0][1]

    def call(self, inputs, **kwargs):
        old_means, old_logvars, new_means, new_logvars = inputs
        log_det_cov_old = K.sum(old_logvars, axis=-1, keepdims=True)
        log_det_cov_new = K.sum(new_logvars, axis=-1, keepdims=True)
        trace_old_new = K.sum(K.exp(old_logvars - new_logvars), axis=-1, keepdims=True)
        kl = 0.5 * (log_det_cov_new - log_det_cov_old + trace_old_new +
                    K.sum(K.square(new_means - old_means) /
                          K.exp(new_logvars), axis=-1, keepdims=True) -
                    np.float32(self.act_dim))
        entropy = 0.5 * (np.float32(self.act_dim) * (np.log(2 * np.pi) + 1.0) +
                         K.sum(new_logvars, axis=-1, keepdims=True))

        return [kl, entropy]


class LogProb(Layer):
    """Layer calculates log probabilities of a batch of actions."""
    def __init__(self, **kwargs):
        super(LogProb, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        actions, act_means, act_logvars = inputs
        logp = -0.5 * K.sum(act_logvars, axis=-1, keepdims=True)
        logp += -0.5 * K.sum(K.square(actions - act_means) / K.exp(act_logvars),
                             axis=-1, keepdims=True)

        return logp


class TRPO(Model):
    # Need to explicitly call Model so that it becomes <tensorflow.python.keras.engine.training.Model object at 0x7faf5712ddd8>
    # Right now, it is <class 'tensorflow.python.keras.engine.training.Model'>
    # Then I should be able to call model.fit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # This is the function that interacts directly with Keras so that
    # all of the Keras functions can be utilized
    def __init__(self, bl,c,layersizes,dropout,deepest,obs_dim, kl_targ, init_logvar, eta, **kwargs):
        super(TRPO, self).__init__(**kwargs)
        self.kl_targ = kl_targ
        self.eta = eta
        self.beta = self.add_weight('beta', initializer='zeros', trainable=False)
        self.policy = PolicyNN(bl,c,layersizes,dropout,deepest,obs_dim, init_logvar)
        self.logprob = LogProb()
        self.kl_entropy = KLEntropy()

    def call(self, inputs):
        obs, act, adv, old_means, old_logvars, old_logp = inputs # array called in Policy to update policy in 'update' 'train_on_batch' and 'predict_on_batch'
        new_means, new_logvars = self.policy(obs) # PolicyNN 'call' func that outputs the new means and log-vars under current policy
        new_logp = self.logprob([act, new_means, new_logvars])
        kl, entropy = self.kl_entropy([old_means, old_logvars,
                                       new_means, new_logvars])
        loss1 = -K.mean(adv * K.exp(new_logp - old_logp))
        loss2 = K.mean(self.beta * kl)
        # TODO - Take mean before or after hinge loss?
        loss3 = self.eta * K.square(K.maximum(0.0, K.mean(kl) - 2.0 * self.kl_targ))
        self.add_loss(loss1 + loss2 + loss3)

        return [kl, entropy]
