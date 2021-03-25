# This script retrains the inception architecture in the multitask learning framework.
# Modifications made by Javad Noorbakhsh (javad.noorbakhsh@gmail.com) to the following script:

# https://github.com/Dataweekends/inception-retrain/blob/master/retrain.py

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow.python.util import compat
import hashlib
import os
import pandas as pd
import numpy as np
from histcnn import util
import tensorflow.compat.v1 as tf
from datetime import datetime
from histcnn import inception_multitasklearning_retrain as incret
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import random
import itertools
from tqdm import tqdm
# from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from histcnn.util import DATA_PATH

def get_total_tfrec_count(tfrecordfiles):
    count_tfrec = lambda tfrecordfile : sum(1 for _ in tf.python_io.tf_record_iterator(tfrecordfile))
    tfrec_counts = [count_tfrec(x) for x in tfrecordfiles]
    return sum(tfrec_counts)

def make_clean_summaries_dir(summaries_dir):
    # Summaries for TensorBoard
    if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
    tf.gfile.MakeDirs(summaries_dir)

def create_trainable_ops(learning_rate, task_class_counts, bottleneck_input,
        onehot_list_tensor, dropout_keep_prob, pos_weight = 1.0, optimizer = 'adam'):
    # Add the new layer that we'll be training.
    (train_step, cross_entropy,
     final_softmax_outputs_list) = incret.add_final_training_ops(
        learning_rate, task_class_counts, bottleneck_input,
        onehot_list_tensor, dropout_keep_prob, pos_weight = pos_weight,
        optimizer = optimizer)

    # Create the operations we need to evaluate the accuracy of our new layer.
    (accuracies_list, predictions_list,
     confusion_matrices_list, final_softmax_outputs_list) = incret.add_evaluation_step(
        final_softmax_outputs_list, onehot_list_tensor)
    return (train_step, cross_entropy, accuracies_list,
            predictions_list, confusion_matrices_list, final_softmax_outputs_list)

def train_multilabel_classification_with_inception_CNN(
    label_names,
    dropout_keep_prob = 0.5,
    nClass = 2, # number of classes
    eval_step_interval=10, # How often to evaluate the training results.
    how_many_training_steps=1000, # How many training steps to run before ending.
    learning_rate=0.01, # How large a learning rate to use when training.
    summaries_dir='/tmp/retrain_logs', # Where to save summary logs for TensorBoard.
    train_batch_size=100, # How many images to train on at a time.
    tfrecordfileslist = None, # path to the tfrecord files that store the input data
    saved_model_path=None, # path+checkpoint prefix for storing the model after training
    optimizer = 'adam',
    resampling_label=None, # the label used for resampling (undersampling)
    class_probs=None, # probabilities of classes (used for resampling)
    pos_weight = 1.0 # weight used for cross entropy balancing
    ):

    # Look at the folder structure, and create lists of all the images.
    tf.reset_default_graph()

    make_clean_summaries_dir(summaries_dir)

    num_tasks = len(label_names)
    task_class_counts = [nClass]*num_tasks
    pbar = util.ProgressBar(how_many_training_steps, step=eval_step_interval)

    alternatelists = lambda list1, list2: [x for y in zip(list1, list2) for x in y]

    (train_bottleneck_input,
    train_onehot_list_tensor, _) = incret.get_batch(tfrecordfileslist, train_batch_size, label_names,
        nClass=nClass, resampling_label=resampling_label, class_probs=class_probs)

    (train_step, cross_entropy, accuracies_list,
            predictions_list, confusion_matrices_list, _ ) = create_trainable_ops(
        learning_rate, task_class_counts, train_bottleneck_input,
        train_onehot_list_tensor, dropout_keep_prob, pos_weight=pos_weight, optimizer=optimizer)

    with tf.Session() as sess:
        # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train',sess.graph)
        # Initialize variables, coordinator and threads
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Train the model
        for i in range(how_many_training_steps):
            train_summary, _ = sess.run([merged, train_step])
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            if (i % eval_step_interval == 0) or (i > how_many_training_steps):
                train_accuracies_list, cross_entropy_value = sess.run([accuracies_list, cross_entropy])
            pbar.Update(("Step {:d} -- Cross entropy = {:f}, Train accuracies : "+"{:s} = {:.2f}, "*num_tasks).format(
                                 i, cross_entropy_value, *alternatelists(label_names, train_accuracies_list)))

        saver = tf.train.Saver()
        saver.save(sess, saved_model_path)

def test_multilabel_classification_with_inception_CNN(
    label_names,
    learning_rate=0.01, # How large a learning rate to use when training.
    nClass = 2, # number of classes
    test_batch_size=-1, # How many images to test on. This test set is only used once, to evaluatethe final accuracy of the model after training completes. A value of -1 causes the entire test set to be used, which leads to more stable results across runs.
    tfrecordfileslist = None, # paths to the tfrecord files that store the input data
    saved_model_path=None, # path for storing the model after training
    avoid_gpu = False,
    optimizer = 'adam',
    pos_weight = 1.0 # weight used for cross entropy balancing
    ):
    dropout_keep_prob = 1

    # Look at the folder structure, and create lists of all the images.
    tf.reset_default_graph()
#     saver = tf.train.import_meta_graph(saved_model_path+'/inception_multitask_retrained.meta')

    num_tasks = len(label_names)
    task_class_counts = [nClass]*num_tasks

    alternatelists = lambda list1, list2: [x for y in zip(list1, list2) for x in y]

    (test_bottleneck_input,
     test_onehot_list_tensor,
     imagefilenames_tensor) = incret.get_batch(tfrecordfileslist, test_batch_size, label_names, nClass= nClass, shuffle_batch=False)

    (train_step, cross_entropy, accuracies_tensor_list, predictions_tensor_list,
        confusion_matrices_tensor_list, final_softmax_outputs_tensor_list) = create_trainable_ops(
        learning_rate, task_class_counts, test_bottleneck_input,
        test_onehot_list_tensor, dropout_keep_prob, pos_weight = pos_weight, optimizer = optimizer)

    if avoid_gpu:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        config = None
    with tf.Session(config=config) as sess:
        # load the trained model
        saver = tf.train.Saver()
        print(saved_model_path)
        saver.restore(sess, tf.train.latest_checkpoint(saved_model_path))

        # Initialize variables, coordinator and threads
        sess.run(tf.local_variables_initializer())

       # Test the trained model
        test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list = sess.run(
           [accuracies_tensor_list, predictions_tensor_list, confusion_matrices_tensor_list, imagefilenames_tensor,
               final_softmax_outputs_tensor_list])
        print(("\nFinal test accuracies : "+"{:s} = {:.2f}, "*num_tasks).format(
            *alternatelists(label_names, test_accuracies_list)))
    return test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list

def run_multilabel_classification_with_inception_CNN(
    label_names, tfrecordfiles_dict, test_batch_size=5000, train_batch_size=100,
    how_many_training_steps=1000, do_not_train = False, eval_step_interval=10,
    nClass = 2, # number of classes
    saved_model_path=None, skip_testing=False, test_on_everything = False,
    summaries_dir='/tmp/retrain_logs', # Where to save summary logs for TensorBoard.
    avoid_gpu_for_testing = False,
    dropout_keep_prob = 0.5,
    optimizer = 'adam',
    resampling_label=None, # the label used for resampling (undersampling)
    class_probs=None, # probabilities of classes (used for resampling)
    pos_weight = 1.0 # weight used for cross entropy balancing
    ):

    assert len(os.path.basename(saved_model_path))>0, 'Model path requires filename prefix to be provided.'
    saved_model_dirname = os.path.dirname(saved_model_path)
    if not os.path.exists(saved_model_dirname):
        os.makedirs(saved_model_dirname)

    if not do_not_train:
        train_multilabel_classification_with_inception_CNN(
            label_names, nClass=nClass, train_batch_size = train_batch_size, dropout_keep_prob = dropout_keep_prob,
            how_many_training_steps=how_many_training_steps, optimizer = optimizer,
            tfrecordfileslist = tfrecordfiles_dict['training'], summaries_dir = summaries_dir,
            saved_model_path=saved_model_path, eval_step_interval=eval_step_interval,
            resampling_label=resampling_label, class_probs=class_probs,
            pos_weight = pos_weight)

    if not skip_testing:
        if test_on_everything:
            test_tfrecordfiles = [x for _, values in tfrecordfiles_dict.items() for x in values]
        else:
            test_tfrecordfiles = tfrecordfiles_dict['testing']
        if test_batch_size=='all':
            test_batch_size = get_total_tfrec_count(test_tfrecordfiles)
        test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list = \
        test_multilabel_classification_with_inception_CNN(
            label_names, nClass=nClass, test_batch_size = test_batch_size,
            tfrecordfileslist = test_tfrecordfiles, optimizer = optimizer,
            saved_model_path = saved_model_dirname,
            avoid_gpu = avoid_gpu_for_testing, pos_weight = pos_weight)

        return test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list
    else:
        return None

def cache_the_uncacheded(image_files_metadata, model_dir = DATA_PATH+'graphs/',
                              use_tqdm_notebook_widget=True):
    from tqdm._tqdm_notebook import tqdm_notebook
    tf.reset_default_graph()
    _, bottleneck_tensor, jpeg_data_tensor, _ = incret.create_inception_graph(model_dir)

    with tf.Session() as sess:
        incret.cache_bottlenecks(sess, image_files_metadata, jpeg_data_tensor, bottleneck_tensor,
                                 use_tqdm_notebook_widget = use_tqdm_notebook_widget)


def test_multilabel_classification_with_inception_CNN_fast(
    image_files_metadata,
    label_names,
    learning_rate=0.01, # How large a learning rate to use when training.
    nClass = 2, # number of classes
    test_batch_size=-1, # How many images to test on. This test set is only used once, to evaluatethe final accuracy of the model after training completes. A value of -1 causes the entire test set to be used, which leads to more stable results across runs.
    tfrecordfileslist = None, # paths to the tfrecord files that store the input data
    saved_model_path=None, # path for storing the model after training
    avoid_gpu = False,
    optimizer = 'adam',
    batch_size = 512,
    eval_step_interval = 10,
    pos_weight = 1.0, # weight used for cross entropy balancing
    quiet_mode = False
    ):

    if quiet_mode:
        tf.logging.set_verbosity(tf.logging.WARN)
    num_tasks = len(label_names)
    assert num_tasks == 1, 'This code does not work with multitask-learning yet'

    dropout_keep_prob = 1

    # Look at the folder structure, and create lists of all the images.
    tf.reset_default_graph()
#     saver = tf.train.import_meta_graph(saved_model_path+'/inception_multitask_retrained.meta')

    task_class_counts = [nClass]*num_tasks

    (test_bottleneck_input,
     test_onehot_list_tensor,
     imagefilenames_tensor) = incret.get_batch(tfrecordfileslist, batch_size, label_names, nClass= nClass, shuffle_batch=False)

    (train_step, _, _, predictions_tensor_list, _, final_softmax_outputs_tensor_list) = create_trainable_ops(
        learning_rate, task_class_counts, test_bottleneck_input,
        test_onehot_list_tensor, dropout_keep_prob, pos_weight = pos_weight, optimizer = optimizer)

    if avoid_gpu:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        config = None

    N = batch_size
    num_steps = int(test_batch_size/N)
    predictions_list = np.nan * np.ones(num_steps*(N+1))
    imagefilenames = np.array([b'']*(num_steps*(N+1)), dtype=object)
    final_softmax_outputs_list = np.nan * np.ones((num_steps*(N+1), 2))

    with tf.Session(config=config) as sess:
        # load the trained model
        saver = tf.train.Saver()
        if not quiet_mode:
            print(saved_model_path)
        saver.restore(sess, tf.train.latest_checkpoint(saved_model_path))

        # Initialize variables, coordinator and threads
        sess.run(tf.local_variables_initializer())

        # Train the model

        pbar = range(num_steps)
        if not quiet_mode:
            pbar = tqdm(pbar)
        for i in pbar:
            predictions_list_, imagefilenames_, final_softmax_outputs_list_ = sess.run(
               [predictions_tensor_list, imagefilenames_tensor, final_softmax_outputs_tensor_list])


            (predictions_list[i*N: (i+1)*N],
             imagefilenames[i*N: (i+1)*N],
             final_softmax_outputs_list[i*N: (i+1)*N, :]) = (predictions_list_[0], imagefilenames_, final_softmax_outputs_list_[0])

    idx = ~np.isnan(predictions_list)
    predictions_list = predictions_list[idx]
    imagefilenames = imagefilenames[idx]
    final_softmax_outputs_list = final_softmax_outputs_list[idx]

    preds = pd.DataFrame({'rel_path': imagefilenames, 'label_pred': predictions_list[0]})
    preds['rel_path'] = preds['rel_path'].map(lambda s: s.decode())
    preds = image_files_metadata.merge(preds, on='rel_path', how='inner')
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(preds['label'], preds['label_pred'])

    return None, [predictions_list], [confusion_matrix], imagefilenames, [final_softmax_outputs_list]
