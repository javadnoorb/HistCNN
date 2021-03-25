from google.cloud import pubsub_v1 as pubsub
from google.cloud import storage
from google.cloud import datastore
import time
import multiprocessing
import pandas as pd
import tensorflow.compat.v1 as tf
from histcnn import (choose_input_list,
                     handle_tfrecords,
                     handle_google_cloud_apis,
                     util,
                     choose_input_list,
                     run_classification,
                     plotting_cnn)
import os
import sys
import re
import glob
import pickle

from histcnn import inception_multitasklearning_retrain as incret

project_id = PROJECT_ID
subscription_name = SUBSCRIPTION_NAME
input_bucket = INPUT_BUCKET
task_kind = TASK_KIND
annotations_path = ANNOTATIONS_PATH
results_path = RESULTS_PATH
pancancer_tfrecords_path = PANCANCER_TFRECORDS_PATH



subscriber = pubsub.SubscriberClient()
subscription_path = subscriber.subscription_path(
    project_id, subscription_name)

NUM_MESSAGES = 1
ACK_DEADLINE = 60
SLEEP_TIME = 30

def mark_done(client, task_id, completed_time, elapsed_time_s):
    with client.transaction():
        key = client.key(task_kind, task_id)
        task = client.get(key)

        if not task:
            raise ValueError('{} {} does not exist.'.format(task_kind, task_id))

        task['status'] = 'Done'
        task['completed_time'] = completed_time
        task['elapsed_time_s'] = elapsed_time_s
        client.put(task)

def mark_in_progress(client, task_id):
    with client.transaction():
        key = client.key(task_kind, task_id)
        task = client.get(key)

        if not task:
            raise ValueError(
                '{} {} does not exist.'.format(task_kind, task_id))

        task['status'] = 'InProgress'
        client.put(task)



def run_tumor_normal_classification(cancertype, how_many_training_steps = 2000, dropout_keep_prob = 0.8, label_names = ['is_tumor'],
                                    optimizer = 'adam', is_weighted = 0, nClass = 2, treat_validation_as_test=True,
                                    do_not_train=True, avoid_gpu_for_testing=True, train_test_percentage = [70, 30]):
# annotations_path='data/pancancer_annotations/tn_frozen_cache_anns'
# results_path = 'data/run-results/frozen_undersampled/'
# pancancer_tfrecords_path='tfrecords/frozen/tn'

    image_file_metadata_filename = '{:s}/{:s}/caches_basic_annotations.txt'.format(annotations_path, cancertype)
#     tfrecords_path = os.path.join(pancancer_tfrecords_path, cancertype, 'caches_512x512/')
    tfrecords_path = os.path.join(pancancer_tfrecords_path, cancertype, '')
    print('copying files from GCS')
    input_bucket_path = 'gs://'+input_bucket+'/'
    util.gsutil_cp(os.path.join(input_bucket_path, tfrecords_path, 'tfrecord*'), '/sdata/'+ tfrecords_path, make_dir=True)
    util.gsutil_cp(os.path.join(input_bucket_path, image_file_metadata_filename), '/sdata/'+ image_file_metadata_filename, make_dir=False)

    # output paths
    trecords_prefix = '/sdata/'+ tfrecords_path + 'tfrecord'
    saved_model_path = os.path.join(results_path, 'saved_models/{:s}'.format(cancertype))
    tensorboard_path = os.path.join(results_path, 'tensorboard_logs/{:s}'.format(cancertype))
    pickle_path = os.path.join(results_path,
                               'pickles/pickles_train{:d}_test{:d}/run_cnn_output_{:s}.pkl'.format(*train_test_percentage, cancertype))

    tfrecordfiles = glob.glob('{:s}*'.format(trecords_prefix))
    assert len(tfrecordfiles)>0
    num_tfrecords = int(len(tfrecordfiles)/3)

    tfrecordfiles_dict = {s: ['{:s}{:d}.{:s}'.format(trecords_prefix, n, s) for n in range(num_tfrecords)] for s in ['training', 'testing', 'validation']}
    image_files_metadata = pd.read_csv('/sdata/' + image_file_metadata_filename, index_col=0)

    if treat_validation_as_test:
        image_files_metadata['crossval_group'].replace('validation', 'testing', inplace=True)
        tfrecordfiles_dict['testing'] = tfrecordfiles_dict['testing']+tfrecordfiles_dict.pop('validation')

    test_batch_size = (image_files_metadata['crossval_group'] == 'testing').sum()

    if is_weighted:
        label_ratio = image_files_metadata[label_names].mean()
        pos_weight = (1/label_ratio - 1).tolist()
    else:
        pos_weight = 1

    class_probs = image_files_metadata.loc[image_files_metadata['crossval_group'] == 'training', label_names[0]].value_counts(normalize=True, sort=False).sort_index().values

    test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list = \
    run_classification.run_multilabel_classification_with_inception_CNN(label_names, tfrecordfiles_dict, test_batch_size=test_batch_size, nClass=nClass,
                                                                        train_batch_size = 512, how_many_training_steps=how_many_training_steps, avoid_gpu_for_testing=avoid_gpu_for_testing,
                                                                        do_not_train = do_not_train, pos_weight = pos_weight, dropout_keep_prob = dropout_keep_prob,
                                                                        saved_model_path = os.path.join('/sdata', saved_model_path, 'mychckpt'),
                                                                        summaries_dir = '/sdata/'+ tensorboard_path, optimizer = optimizer,
                                                                        class_probs=class_probs)

    util.mkdir_if_not_exist(os.path.dirname('/sdata/' + pickle_path))

    pickle.dump([image_files_metadata, test_accuracies_list, predictions_list,
                 confusion_matrices_list, imagefilenames, final_softmax_outputs_list],
                open('/sdata/' + pickle_path, 'wb'))

    util.gsutil_cp(os.path.join('/sdata', saved_model_path), os.path.join(input_bucket_path, saved_model_path))
    util.gsutil_cp(os.path.join('/sdata', tensorboard_path), os.path.join(input_bucket_path, tensorboard_path))
    util.gsutil_cp(os.path.join('/sdata', pickle_path), os.path.join(input_bucket_path, pickle_path))


def worker(msg):
    start_time = time.time()
    print(msg.message.data)

    task_id = int(msg.message.data)
    client = datastore.Client(project_id)
    key = client.key(task_kind, task_id)
    params = client.get(key)

    # Setting the status to 'InProgress'
    mark_in_progress(client, task_id)

    cancertype = params['cancertype']

    label_names = ['is_tumor']

    run_tumor_normal_classification(cancertype, label_names = label_names, treat_validation_as_test=True, do_not_train=False)

    elapsed_time_s = round((time.time() - start_time), 1)  # in seconds
    completed_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # We now can comfirm the job
    client = datastore.Client(project_id)
    mark_done(client=client, task_id=task_id, completed_time=completed_time,
              elapsed_time_s=elapsed_time_s)

    print('Finish Timestamp: {} - Time elapsed: {} seconds.'.format(completed_time, elapsed_time_s))

    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_name)

    # Acknowledging the message
    subscriber.acknowledge(subscription_path, [msg.ack_id])
    print("{}: Acknowledged {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg.message.data))


# The subscriber pulls a specific number of messages.
response = subscriber.pull(subscription_path, max_messages=NUM_MESSAGES)

# `processes` stores process as key and ack id and message as values.
processes = dict()
for message in response.received_messages:
    process = multiprocessing.Process(target=worker, args=(message,))
    processes[process] = (message.ack_id, message.message.data)
    process.start()

while processes:
    for process in list(processes):
        ack_id, msg_data = processes[process]
        # If the process is still running, reset the ack deadline as
        # specified by ACK_DEADLINE once every while as specified
        # by SLEEP_TIME.
        if process.is_alive():
            # `ack_deadline_seconds` must be between 10 to 600.
            subscriber.modify_ack_deadline(
                subscription_path,
                [ack_id],
                ack_deadline_seconds=ACK_DEADLINE)
            print('{}: Reset ack deadline for {} for {}s'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                msg_data, ACK_DEADLINE))

        # If the processs is finished, acknowledges using `ack_id`.
        else:
            #subscriber.acknowledge(subscription_path, [ack_id])
            processes.pop(process)

    # If there are still processes running, sleeps the thread.
    if processes:
        time.sleep(SLEEP_TIME)
