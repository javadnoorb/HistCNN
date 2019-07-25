from google.cloud import pubsub_v1 as pubsub
from google.cloud import storage
from google.cloud import datastore
import time
import multiprocessing
import logging
import pandas as pd
import tensorflow as tf
from histcnn import (choose_input_list,
                     handle_tfrecords,
                     handle_google_cloud_apis,
                     util)
import os
import sys
import re
from tqdm import tqdm

project_id = PROJECT_ID
subscription_name = SUBSCRIPTION_NAME
tiles_input_bucket = TILES_INPUT_BUCKET
task_kind = TASK_KIND
gcs_ann_path = GCS_ANN_PATH

subscriber = pubsub.SubscriberClient()
subscription_path = subscriber.subscription_path(
    project_id, subscription_name)

NUM_MESSAGES = 1
ACK_DEADLINE = 60
SLEEP_TIME = 30

def mark_done(client, task_id, completed_time, elapsed_time_s,
              shard_length_tiles, tfrecord_size_MBi):
    with client.transaction():
        key = client.key(task_kind, task_id)
        task = client.get(key)

        if not task:
            raise ValueError('{} {} does not exist.'.format(task_kind, task_id))

        task['status'] = 'Done'
        task['completed_time'] = completed_time
        task['elapsed_time_s'] = elapsed_time_s
        task['shard_length_tiles'] = shard_length_tiles
        task['tfrecord_size_MBi'] = tfrecord_size_MBi
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
    category = params['category']
    shard_length = int(params['shard_length'])
    shard_index = int(params['shard_index'])
    gcs_output_path = params['gcs_output_path']

    print('Loading metadata...')
    image_file_metadata_filename = 'data/caches_basic_annotations.txt'
    util.gsutil_cp('{}/{}/caches_basic_annotations.txt'.format(gcs_ann_path, cancertype), 'data/', make_dir=True)
    image_files_metadata = pd.read_csv(image_file_metadata_filename, skiprows=range(1, shard_index*shard_length+1), nrows=shard_length)

    shard_length_tiles = len(image_files_metadata.index)

    label_names = ['is_tumor']

    print('Downloading cache files...')
    image_files_metadata['cache_values'] = choose_input_list.load_cache_values(image_files_metadata, 
                                                                               bucket_name = tiles_input_bucket,
                                                                               notebook = False)
    
#    print('Downloading tiles...')
#    bucket = handle_google_cloud_apis.gcsbucket(project_id, tiles_input_bucket)
#    def download_tile(df_row, bucket):
#        gcs_rel_path = df_row['GCSurl'][len('gs://' + bucket.bucket_name)+1:]
#        bucket.download_from_gcs(gcs_rel_path, output_dir=df_row['rel_path'])

#    tqdm.pandas()
#    image_files_metadata.progress_apply(lambda df_row: download_tile(df_row, bucket), axis=1)

    crossval_groups = ['training','testing','validation']
    if category not in crossval_groups+['all']:
        raise Exception('Unknown cross validation category.')

    # Create tfrecords for each category
    if category != 'all': # keyword 'all' will loop through all three categories
        crossval_groups = [category]

    tfrecords_folder = 'tfrecords_{}'.format(cancertype)
    util.mkdir_if_not_exist(tfrecords_folder)

    for category in crossval_groups:
        print('Creating TFRecord for {:s}...'.format(category))
        handle_tfrecords.create_tfrecords_per_category_for_caches(image_files_metadata, label_names, category,
                                                                 tfrecord_file_name_prefix = tfrecords_folder + '/tfrecord{:d}'.format(shard_index))

    tfrecords_bucket = re.search('gs://(.+?)/', gcs_output_path).group(1)
    prefix = 'gs://' + tfrecords_bucket + '/'
    gcs_directory = "".join(gcs_output_path.rsplit(prefix))

    bucket = handle_google_cloud_apis.gcsbucket(project_name=project_id, bucket_name=tfrecords_bucket)
    bucket.copy_files_to_gcs(tfrecords_folder, gcs_directory, verbose=True)

    command = 'du -s ' + tfrecords_folder + '/'
    tfrecord_size_MBi = round(int(os.popen(command).read().split()[0])/1000,1)  # in MB

    # Removing local files
    command = "rm -rf " + tfrecords_folder
    os.popen(command)
    os.popen("rm -rf tcga_tiles")

    elapsed_time_s = round((time.time() - start_time), 1)  # in seconds

    completed_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # We now can comfirm the job
    client = datastore.Client(project_id)
    mark_done(client=client, task_id=task_id, completed_time=completed_time,
              elapsed_time_s=elapsed_time_s, shard_length_tiles=shard_length_tiles,
              tfrecord_size_MBi=tfrecord_size_MBi)

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

print("Received and acknowledged {} messages. Done.".format(NUM_MESSAGES))
