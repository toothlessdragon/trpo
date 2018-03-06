import numpy as np
import pandas as pd
import tensorflow as tf
import os



# ENTER LIST OF LOG FILENAMES HERE:
filepaths = ['log-files/Humanoid-v2/Feb-26_02.01.28']

for filepath in filepaths:
    tbwriter = tf.summary.FileWriter(filepath)

    path = os.path.join(filepath, 'log.csv')
    data = pd.read_csv(path)

    for _, row in data.iterrows():

        episode = row["_Episode"]

        summary = [tf.Summary.Value(tag=tag, simple_value=val)
                   for tag, val in row.iteritems()
                   if tag is not "_Episode"]
        tbwriter.add_summary(tf.Summary(value=summary), episode)