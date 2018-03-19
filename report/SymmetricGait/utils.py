"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os
import shutil
import glob
import csv
import tensorflow as tf


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)


    def update(self, x, state):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
            state: tensorflow state storage

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        means, vars, m = state.get_scaler()

        if m <= 0:
            means = np.mean(x, axis=0)
            vars = np.var(x, axis=0)
            m = x.shape[0]
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((means * m) + (new_data_mean * n)) / (m + n)
            vars = (((m * (vars + np.square(means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (m + n) -
                         np.square(new_means))
            vars = np.maximum(0.0, vars)  # occasionally goes negative, clip
            means = new_means
            m += n

        self.means = means
        self.vars = vars
        state.update_scaler(means, vars, m)

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, sub_dir):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            sub_dir: unique sub-directory name (e.g. date/time string)
        """
        self.write_header = False
        path = os.path.join('log-files', logname, sub_dir)
        if not os.path.exists(path):
            os.makedirs(path)
            self.write_header = True
        filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        for filename in filenames:     # for reference
            shutil.copy(filename, path)
        self.tbwriter = tf.summary.FileWriter(path)
        path = os.path.join(path, 'log.csv')

        self.init = False
        self.log_entry = {}
        self.f = open(path, 'a')
        self.writer = None  # DictWriter created with first call to write() method

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)
        if not self.init:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.init = True
        if self.write_header:
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)

        episode = self.log_entry["_Episode"]
        summary = [tf.Summary.Value(tag=tag, simple_value=val)
                   for tag, val in self.log_entry.items()
                   if tag is not "_Episode"]

        self.tbwriter.add_summary(tf.Summary(value=summary), episode)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                               log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()
