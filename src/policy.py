"""
NN Policy with KL Divergence Constraint (PPO / TRPO)

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow


class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, weights_path):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
            weights_path: path of weights to load
        """
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.epochs = 20
        self.raw_lr = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()
        if weights_path is not None:
            self.restore(weights_path)

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._scaler()
            self._policy_nn()
            self._dynamic_hyperparameters()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=1)

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # # strength of D_KL loss terms:
        # self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        # self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # # learning rate:
        # self.lr_ph = tf.placeholder(tf.float32, (), 'learningrate')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _dynamic_hyperparameters(self, beta=1.0, eta=50.0, lr_multiplier=1.0):
        """
        Initalize dynamic hyperparameters

        beta:  dynamically adjusted D_KL loss multiplier
        eta:  multiplier for D_KL-kl_targ hinge-squared loss
        lr_multiplier: dynamically adjust lr when D_KL out of control
        """
        # strength of D_KL loss terms:
        self.beta = tf.get_variable("beta", dtype=tf.float32, initializer=tf.constant(beta), trainable=False)
        self.eta = tf.get_variable("eta", dtype=tf.float32, initializer=tf.constant(eta), trainable=False)
        # learning rate multiplier:
        self.lr_multiplier = tf.get_variable("lr_mult", dtype=tf.float32,
                                             initializer=tf.constant(lr_multiplier),
                                             trainable=False)
        self.lr = self.raw_lr * self.lr_multiplier


    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.raw_lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined

        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="means")
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.raw_lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        loss1 = -tf.reduce_mean(self.advantages_ph *
                                tf.exp(self.logp - self.logp_old))
        loss2 = tf.reduce_mean(self.beta * self.kl)
        loss3 = self.eta * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
        self.loss = loss1 + loss2 + loss3
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        beta, lr_multiplier = self.sess.run([self.beta, self.lr_multiplier])
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            beta = np.minimum(35, 1.5 * beta)  # max clip beta
            if beta > 30 and lr_multiplier > 0.1:
                lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            beta = np.maximum(1 / 35, beta / 1.5)  # min clip beta
            if beta < (1 / 30) and lr_multiplier < 10:
                lr_multiplier *= 1.5
        self.sess.run([self.beta.assign(beta), self.lr_multiplier.assign(lr_multiplier)])

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': beta,
                    '_lr_multiplier': lr_multiplier})


    def _scaler(self):
        ''' Scaler implementation in tensorflow'''
        self.obs_means = tf.get_variable("obs_mean", shape=[self.obs_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                                    trainable=False)
        self.vars = tf.get_variable("var", shape=[self.obs_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                                   trainable=False)

        self.m = tf.get_variable("obs_count", dtype=tf.float32, initializer=tf.constant(0.0), trainable=False)

    def get_scaler(self):
        return self.sess.run([self.obs_means, self.vars, self.m])

    def update_scaler(self, means, var, m):
        self.sess.run([self.obs_means.assign(means), self.vars.assign(var), self.m.assign(m)])

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def save(self, path, step):
        """ Save all weights to path """
        save_path = self.saver.save(self.sess, os.path.join(path, "weights"), global_step=step)
        print("Model saved in path: %s" % save_path)

    # def restore(self, path):
     #   """ Restore weights from path """
      #  self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
       # print("Model restored.")

    def restore(self, path):
        """ Restore weights from path """
        var_list= self.print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(path))
        variables = self.g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        restore_list = [var for var in variables if var._shared_name in var_list]
        saver = tf.train.Saver(restore_list)
        saver.restore(self.sess, tf.train.latest_checkpoint(path))
        print("Model restored.")


    def print_tensors_in_checkpoint_file(self, file_name):
        varlist=[]
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
        return varlist

