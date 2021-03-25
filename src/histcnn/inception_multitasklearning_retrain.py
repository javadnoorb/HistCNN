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
"""
This code takes Inception v3 architecture model trained on
ImageNet images, and trains a new top layer plus extra layers for multitask training.

To use with TensorBoard:       tensorboard --logdir /tmp/retrain_logs

"""

import hashlib
import os.path
import random
import re
import sys
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from tqdm._tqdm_notebook import tqdm_notebook
from tqdm import tqdm

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
SHARED_LAYER_OUTPUT_SIZE  = 1024
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
IMAGE_EXTENSIONS = ['jpg']
BOTTLENECK_DATAFRAME_KEYWORD = 'rel_path'
IMAGE_DATAFRAME_KEYWORD = 'image_filename'
NUM_THREADS = 16
SHUFFLE_BUFFER_SIZE = 1000


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
        dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def resize_and_convert_tiff(infile,outfile,image_size):
    from PIL import Image
    im = Image.open(infile)
    im.thumbnail(image_size, Image.ANTIALIAS)
    im.save(outfile, "JPEG", quality=100)

def maybe_resize_and_convert_all_tiff(tiff_dir, out_dir, image_size):
    import glob
    infiles = glob.glob(tiff_dir+'*/*.tif')
    for infile in infiles:
        infile_split = infile.split('/')
        sub_dir_path = os.path.join(out_dir,infile_split[-2])
        ensure_dir_exists(sub_dir_path)
        outfile_base = os.path.splitext(infile_split[-1])[0]+'.jpg'
        outfile = os.path.join(sub_dir_path,outfile_base)
        if not os.path.exists(outfile):
            sys.stdout.write('\rCreating JPEG file at ' + outfile_base+' '*20)
            sys.stdout.flush()
            resize_and_convert_tiff(infile,outfile,image_size)

def maybe_download_and_extract(dest_directory):
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    from six.moves import urllib
    import tarfile

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %(filename,float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,filepath,_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.\n')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_inception_graph(model_dir):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    with tf.Session() as sess:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))
        return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
        sess: Current active TensorFlow Session.
        image_data: String of raw JPEG data.
        image_data_tensor: Input data layer in the graph.
        bottleneck_tensor: Layer before the final softmax.

    Returns:
        Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor: image_data})
    return np.squeeze(bottleneck_values)

def get_or_create_bottleneck(sess, image_files_metadata_row, jpeg_data_tensor, bottleneck_tensor):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
        sess: The current active TensorFlow Session.
        image_files_metadata_row: row of the dataframe containing image metadata
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        bottleneck_tensor: The output tensor for the bottleneck values.

    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    ensure_dir_exists(os.path.dirname(image_files_metadata_row[BOTTLENECK_DATAFRAME_KEYWORD]))
    is_cached = os.path.isfile(image_files_metadata_row[BOTTLENECK_DATAFRAME_KEYWORD])
    if is_cached:
        with open(image_files_metadata_row[BOTTLENECK_DATAFRAME_KEYWORD], 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
    if not is_cached or bottleneck_string=='':
        assert os.path.exists(image_files_metadata_row[IMAGE_DATAFRAME_KEYWORD]),"{} does not exist.".format(image_files_metadata_row[IMAGE_DATAFRAME_KEYWORD])
        image_data = gfile.FastGFile(image_files_metadata_row[IMAGE_DATAFRAME_KEYWORD], 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(image_files_metadata_row[BOTTLENECK_DATAFRAME_KEYWORD], 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    assert len(bottleneck_values)==BOTTLENECK_TENSOR_SIZE,"Cache file {:s} only has {:d} elements, but expected {:d}.".format(image_files_metadata_row[BOTTLENECK_DATAFRAME_KEYWORD],len(bottleneck_values),BOTTLENECK_TENSOR_SIZE)
    return bottleneck_values

def cache_bottlenecks(sess, image_files_metadata, jpeg_data_tensor, bottleneck_tensor,use_tqdm_notebook_widget=True):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
        sess: The current active TensorFlow Session.
        image_files_metadata: dataframe of training images for each label.
        jpeg_data_tensor: Input tensor for jpeg data from file.
        bottleneck_tensor: The penultimate output layer of the graph.
    """

    # still not sure how robust tqdm is. Maybe will use the 'old code'
    if use_tqdm_notebook_widget:
        tqdm_notebook.pandas(desc='Caching...')
    else:
        tqdm.pandas(desc='Caching...')
    alreadycached_first = image_files_metadata[BOTTLENECK_DATAFRAME_KEYWORD].apply(
        os.path.isfile).sort_values(ascending=False).index # this ensures that first the function will go through files that are already cached, so that the progress bar doesn't jump back and forth between slow and fast mode
    image_files_metadata.loc[alreadycached_first].progress_apply(
        lambda image_files_metadata_row: get_or_create_bottleneck(
            sess, image_files_metadata_row, jpeg_data_tensor, bottleneck_tensor), axis=1)

def get_random_cached_bottlenecks(sess, image_files_metadata, label_names,
                                  how_many, category, jpeg_data_tensor, bottleneck_tensor, n_class=2 ):
    """Retrieves bottleneck values for cached images.

    This function can retrieve the cached bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
        sess: Current TensorFlow Session.
        image_files_metadata: dataframe of image metadata
        how_many: If positive, a random sample of this size will be chosen.
        If negative, all bottlenecks will be retrieved.
        category: Name string of which set to pull from - training, testing, or
        validation.
        jpeg_data_tensor: The layer to feed jpeg image data into.
        bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
        List of bottleneck arrays, their corresponding ground truths, and the
        relevant filenames.
    """
    bottlenecks = []
    filenames = []

    if how_many < 0:
        image_files_metadata_batch = image_files_metadata[image_files_metadata['crossval_group'] == category]
    else:
        image_files_metadata_batch = image_files_metadata[image_files_metadata['crossval_group'] == category].sample(n=how_many)
    for _,image_files_metadata_row in image_files_metadata_batch.iterrows():
        # Retrieve a random sample of bottlenecks.
        bottleneck = get_or_create_bottleneck(sess, image_files_metadata_row, jpeg_data_tensor,bottleneck_tensor)
        bottlenecks.append(bottleneck)
        filenames.append(image_files_metadata_row[IMAGE_DATAFRAME_KEYWORD])
    labels_list = image_files_metadata_batch[label_names].T.values.tolist()
    labels_onehot_list = sess.run(tf.one_hot(labels_list,n_class))
    return bottlenecks, labels_onehot_list, filenames

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def _parse_example(example, label_names, nClass = 2):
    feature = {'sampleid': tf.FixedLenFeature([], tf.string),
               'cachefile': tf.FixedLenFeature([], tf.string),
               'image': tf.FixedLenFeature([], tf.string)}
    label_features = {label_name: tf.FixedLenFeature([], tf.int64)
                      for label_name in label_names}
    feature.update(label_features)
    parsed_example = tf.parse_single_example(example, features=feature)
    return parsed_example

def undersampling_filter(example, class_probs, label):
    """
    adapted from https://stackoverflow.com/questions/47236465/oversampling-functionality-in-tensorflow-dataset-api
    Computes if given example is rejected or not.
    """
    assert sum(class_probs) == 1
    class_prob = tf.gather(tf.constant(class_probs), example[label])
    prob_ratio = tf.cast(min(class_probs)/class_prob, dtype=tf.float32)
    prob_ratio = tf.minimum(prob_ratio, 1.0)

    acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), prob_ratio)
    return acceptance

def _extract_features(parsed_example, label_names, nClass = 2):
    # parsed_example = _parse_example(example, label_names, nClass = nClass)
    image = tf.decode_raw(parsed_example['image'], tf.float32)
    labels = [tf.stack(tf.one_hot(tf.cast(parsed_example[label_name], tf.int32), nClass))
                  for label_name in label_names]
    imagefilename = tf.cast(parsed_example['cachefile'], tf.string)
    return image, labels, imagefilename

def _get_batch(tfrecordfileslist, batch_size, label_names, nClass=2, shuffle_batch=True, resampling_label=None, class_probs=None):
    dataset = tf.data.TFRecordDataset(tfrecordfileslist)
    dataset = dataset.map(lambda x:_parse_example(x, label_names, nClass=nClass), num_parallel_calls=NUM_THREADS)
    if class_probs is not None:
        if resampling_label is None:
            resampling_label = label_names[0]
        dataset = dataset.filter(lambda x: undersampling_filter(x, class_probs, resampling_label))
    dataset = dataset.map(lambda x:_extract_features(x, label_names, nClass=nClass), num_parallel_calls=NUM_THREADS)
    dataset = dataset.repeat(-1)
    if shuffle_batch:
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image, labels, imagefilename = iterator.get_next()
    return image, labels, imagefilename

def get_batch(tfrecordfileslist, batch_size, label_names, nClass=2, shuffle_batch=True, resampling_label=None, class_probs=None):

    images, labels, imagefilenames = _get_batch(tfrecordfileslist, batch_size, label_names, nClass=nClass,
                                    shuffle_batch=shuffle_batch, resampling_label=resampling_label,
                                    class_probs=class_probs)

    with tf.name_scope('input'):
        bottlenecks_tensor = tf.squeeze(images, name='bottleneck_input')
        onehot_list_tensor = tf.unstack(labels, axis=1, num = len(label_names),
                                        name = 'labels_onehot_list')

    return bottlenecks_tensor, onehot_list_tensor, imagefilenames


def add_final_training_ops(learning_rate, task_class_counts, bottleneck_input,
                           labels_onehot_placeholders_list, dropout_keep_prob, optimizer = 'adam',
                           pos_weight = 1.0):
    num_tasks = len(task_class_counts)

    with tf.name_scope('custom_final_layer'):
        with tf.name_scope('shared_layer'):
            with tf.name_scope('weights'):
                shared_layer_weights = tf.Variable(tf.truncated_normal(
                    [BOTTLENECK_TENSOR_SIZE, SHARED_LAYER_OUTPUT_SIZE], stddev=0.001),name='shared_layer_weights')
                variable_summaries(shared_layer_weights)
            with tf.name_scope('biases'):
                shared_layer_biases = tf.Variable(tf.zeros([SHARED_LAYER_OUTPUT_SIZE]), name='shared_layer_biases')
                variable_summaries(shared_layer_biases)
            with tf.name_scope('Wx_plus_b'):
                try:
                    shared_layer_output = tf.nn.relu(tf.matmul(bottleneck_input, shared_layer_weights) + shared_layer_biases)
                except:
                    print(bottleneck_input.shape)
                    print(shared_layer_weights.shape)
                    raise
                tf.summary.histogram('shared_layer_output', shared_layer_output)
            with tf.name_scope('dropout'):
                shared_layer_output_dropout = tf.nn.dropout(shared_layer_output, rate=1-dropout_keep_prob,
                                                            name='SharedLayerOutputDroppedout')

        final_softmax_outputs_list = [None]*num_tasks
        cross_entropies_list = [None]*num_tasks
        layer_weights_list = [None]*num_tasks

        if np.isscalar(pos_weight):
            pos_weights = [pos_weight]*num_tasks
        elif hasattr(pos_weight, "__len__") and len(pos_weight)==num_tasks:
            pos_weights = pos_weight
        else:
            assert 'Bad weights input for cross entropy'

        for n in range(num_tasks):
            with tf.name_scope('task%d'%n):
                with tf.name_scope('weights'):
                    layer_weights = tf.Variable(tf.truncated_normal([SHARED_LAYER_OUTPUT_SIZE, task_class_counts[n]], stddev=0.001),
                                                name='layer_weights')
                    variable_summaries(layer_weights)
                with tf.name_scope('biases'):
                    layer_biases = tf.Variable(tf.zeros([task_class_counts[n]]), name='layer_biases')
                    variable_summaries(layer_biases)
                with tf.name_scope('logits'):
                    logits = tf.matmul(shared_layer_output_dropout, layer_weights) + layer_biases
                    tf.summary.histogram('logits', logits)
                final_softmax_outputs_list[n] = tf.nn.softmax(logits, name='final_softmax_output')
                tf.summary.histogram('final_softmax_output', final_softmax_outputs_list[n])

                cross_entropies_list[n] = tf.nn.weighted_cross_entropy_with_logits(
                    labels_onehot_placeholders_list[n], logits, pos_weights[n])

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.add_n(cross_entropies_list)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizers_dict = {'adam': tf.train.AdamOptimizer(learning_rate=learning_rate),
                           'graddesc': tf.train.GradientDescentOptimizer(learning_rate),
                           'rmsprop': tf.train.RMSPropOptimizer(learning_rate)}

        train_step = optimizers_dict[optimizer].minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, final_softmax_outputs_list)

def add_evaluation_step(final_softmax_outputs_list, labels_onehot_placeholders_list):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
        final_tensor: The new final node that produces results.
        labels_placeholders_list: The node we feed ground truth data into.

    Returns:
        Tuple of (evaluation step, prediction, confusion).
    """

    assert type(final_softmax_outputs_list)==list, "A list of final tensors should be provided."
    num_tasks = len(final_softmax_outputs_list)
    predictions_list = [None]*num_tasks
    accuracies_list = [None]*num_tasks
    confusion_matrices_list = [None]*num_tasks

    for n in range(num_tasks):
        with tf.name_scope('task%d'%n):
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    predictions_list[n] = tf.argmax(final_softmax_outputs_list[n], 1)

                    #from IPython.core.debugger import set_trace; set_trace()
                    correct_prediction = tf.equal(predictions_list[n], tf.argmax(labels_onehot_placeholders_list[n], 1))
                with tf.name_scope('accuracy'):
                    accuracies_list[n] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tf.summary.scalar('accuracy', accuracies_list[n])
            with tf.name_scope('confusion'):
                confusion_matrices_list[n] = tf.confusion_matrix(labels=tf.argmax(labels_onehot_placeholders_list[n], 1),
                                                                 predictions=predictions_list[n],
                                           num_classes=labels_onehot_placeholders_list[n].shape.as_list()[1])

    return accuracies_list, predictions_list, confusion_matrices_list, final_softmax_outputs_list
