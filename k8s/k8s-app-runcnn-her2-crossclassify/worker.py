from google.cloud import pubsub_v1 as pubsub
from google.cloud import storage
from google.cloud import datastore
import time
import multiprocessing
import logging
import pandas as pd
import tensorflow as tf
import pickle
from histcnn import (choose_input_list,
                     handle_tfrecords,
                     handle_google_cloud_apis,
                     util,
                     run_classification,
                     plotting_cnn)
import os
import sys
import re
from tqdm import tqdm
import json
import glob

project_id = PROJECT_ID
payer_project_id = PAYER_PROJECTID
subscription_name = SUBSCRIPTION_NAME
input_bucket = INPUT_BUCKET
task_kind = TASK_KIND
annotations_path = ANNOTATIONS_PATH
results_path = RESULTS_PATH
pancancer_tfrecords_path = PANCANCER_TFRECORDS_PATH
gcs_output_path = GCS_OUTPUT_PATH

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

def cross_classify(cancertype1, cancertype2, include_training_set=True, train_test_percentage = [70, 30], 
                   label_terms=['normal', 'tumor'], labal_names = ['cnv'], nClass = 2):
    if cancertype1 == cancertype2:
        include_training_set = False
    print('cross classify {:s} and {:s} ...'.format(cancertype1, cancertype2))
    

    saved_model_path = os.path.join('/sdata', results_path, 'saved_models', cancertype1, '')
    tfrecpath = os.path.join('/sdata', pancancer_tfrecords_path, cancertype2, '')  
    image_file_metadata_filename = os.path.join('/sdata', annotations_path, cancertype2, 'caches_basic_annotations.txt')

    print('copying files from GCS')
    util.gsutil_cp('gs://'+input_bucket+'/'+saved_model_path[len('/sdata/'):]+'*', saved_model_path, make_dir=True, payer_project_id=payer_project_id)
    util.gsutil_cp('gs://'+input_bucket+'/'+tfrecpath[len('/sdata/'):]+'*.testing', tfrecpath, make_dir=True, payer_project_id=payer_project_id)
    util.gsutil_cp('gs://'+input_bucket+'/'+tfrecpath[len('/sdata/'):]+'*.validation', tfrecpath, make_dir=True, payer_project_id=payer_project_id)
    if include_training_set:
        util.gsutil_cp('gs://'+input_bucket+'/'+tfrecpath[len('/sdata/'):]+'*.training', tfrecpath, make_dir=True, payer_project_id=payer_project_id)
    util.gsutil_cp('gs://'+input_bucket+'/'+image_file_metadata_filename[len('/sdata/'):], image_file_metadata_filename, make_dir=False, payer_project_id=payer_project_id)

    image_files_metadata = pd.read_csv(image_file_metadata_filename, sep=',')
    image_files_metadata.rename(columns={'cnv':'label'}, inplace=True)
    image_files_metadata['label_name'] = image_files_metadata['label'].map(lambda x: label_terms[x])

    tfrecordfileslist = glob.glob(tfrecpath+'*.testing') + glob.glob(tfrecpath+'*.validation')
    if include_training_set:
        tfrecordfileslist += glob.glob(tfrecpath+'*.training')
    else:
        image_files_metadata = image_files_metadata[image_files_metadata['crossval_group']!='training']

#     test_batch_size = run_classification.get_total_tfrec_count(tfrecordfileslist)
    test_batch_size = len(image_files_metadata)

    
    
    print('running the CNN')

    test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list = \
    run_classification.test_multilabel_classification_with_inception_CNN_fast(image_files_metadata, labal_names, tfrecordfileslist = tfrecordfileslist,
                                                      saved_model_path = saved_model_path, nClass = nClass, test_batch_size = test_batch_size)

    print('caclulating the AUC')
    votes, predictions_df = plotting_cnn.get_per_slide_average_predictions(image_files_metadata, imagefilenames, predictions_list, ['label'])
    
    roc_auc = {}
    roc_auc['perslide'] = plotting_cnn.plot_perslide_roc(predictions_df, plot_results=False)
    
    roc_auc['pertile'] = plotting_cnn.plot_pertile_roc(imagefilenames, predictions_list, final_softmax_outputs_list, image_files_metadata, plot_results=False)
    
    print('storing the output')
    jsonfile = 'roc_auc_{:s}_{:s}.json'.format(cancertype1, cancertype2)	
    json.dump(roc_auc, open(jsonfile, 'w'))

    util.gsutil_cp(jsonfile, gcs_output_path, payer_project_id=payer_project_id)

    # save output to pickle file
    pickle_dir = os.path.join(results_path, 'pickles/pickles_train{:d}_test{:d}_pickled/'.format(*train_test_percentage))    
    pickle_path = os.path.join(pickle_dir, 'run_cnn_output_{:s}_{:s}.pkl'.format(cancertype1, cancertype2))
    
    util.mkdir_if_not_exist('/sdata/' + pickle_dir)
    pickle.dump([image_files_metadata, test_accuracies_list, predictions_list, 
                 confusion_matrices_list, imagefilenames, final_softmax_outputs_list], 
                open('/sdata/' + pickle_path, 'wb'))

    util.gsutil_cp(os.path.join('/sdata', pickle_path), os.path.join('gs://'+input_bucket, pickle_path), payer_project_id=payer_project_id)


def worker(msg):
    start_time = time.time()
    print(msg.message.data)

    task_id = int(msg.message.data)
    client = datastore.Client(project_id)
    key = client.key(task_kind, task_id)
    params = client.get(key)

    # Setting the status to 'InProgress'
    mark_in_progress(client, task_id)

    cancertype1 = params['cancertype1']
    cancertype2 = params['cancertype2']
	
    cross_classify(cancertype1, cancertype2, label_terms=['not amplified', 'amplified'])

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


