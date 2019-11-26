from google.cloud import pubsub_v1 as pubsub
from google.cloud import storage
from google.cloud import datastore
import time
import multiprocessing
import logging
import matplotlib
matplotlib.use('pdf')
import pandas as pd
from histcnn import (process_files,
                     run_classification,
                     util,
                     handle_google_cloud_apis)
import os
import sys
import re
import glob

project_id = PROJECT_ID
subscription_name = SUBSCRIPTION_NAME
input_bucket_name = INPUT_BUCKET_NAME
output_bucket_name = OUTPUT_BUCKET_NAME
task_kind = TASK_KIND

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
            raise ValueError(
                '{} {} does not exist.'.format(task_kind, task_id))

        task['status'] = 'Done'
        task['completed_time'] = completed_time
        task['elapsed_time_s'] = elapsed_time_s
        task['input_bucket_name'] = input_bucket_name
        task['output_bucket_name'] = output_bucket_name
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
    svs_path = params['svs_path']

#    input_bucket_name = svs_path.lstrip('gs://').split('/')[0]
    input_bucket_path = 'gs://' + input_bucket_name
    output_bucket_path = 'gs://' + output_bucket_name
    input_tiles_path = os.path.join(svs_path, 'tiles/tile_*.jpg')
    local_tiles_path = os.path.join(re.sub(input_bucket_path, '/sdata', svs_path), 'tiles/')
    local_tiles_glob_path = os.path.join(local_tiles_path, 'tile_*.jpg')
#     output_cache_path = re.sub('/tiles_', '/caches_', svs_path)
    x = local_tiles_path.rstrip('/').split('/'); x.pop(-1); x[-2]+='_cache'
    local_cache_path = '/'.join(x)
#     local_cache_path = re.sub(bucket_path, '/sdata', output_cache_path)
    output_cache_path = re.sub('/sdata', output_bucket_path, local_cache_path)
    
    print('copying files from GCS')
    util.gsutil_cp(input_tiles_path, local_tiles_path, make_dir=True)    

    caches_metadata = pd.DataFrame(glob.glob(local_tiles_glob_path), columns=['image_filename'])
    def convert_to_cache_path(x, local_cache_path):
        return os.path.join(local_cache_path, x.split('/')[-1]+'_cached.txt')

    caches_metadata['rel_path'] = caches_metadata['image_filename'].map(lambda x: convert_to_cache_path(x, local_cache_path))
    
    print('cache the unchached...')
    graphs_folder = util.DATA_PATH + 'graphs/'
    run_classification.cache_the_uncacheded(caches_metadata,
                                            model_dir = graphs_folder,
                                            use_tqdm_notebook_widget=False)
    print('Finished caching {:s}'.format(svs_path))    

    print('Copying files from disk to gcs...')
    util.gsutil_cp(os.path.join(local_cache_path, '*'), output_cache_path)

    # Calculate elapsed time
    elapsed_time_s = round((time.time() - start_time), 1)  # in seconds
    completed_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    client = datastore.Client(project_id)
    mark_done(client, task_id, completed_time, elapsed_time_s)

    print('Completed caching of SVS file: {}'.format(svs_path))
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
        else:
            processes.pop(process)

    # If there are still processes running, sleeps the thread.
    if processes:
        time.sleep(SLEEP_TIME)

print("Received and acknowledged {} messages. Done.".format(NUM_MESSAGES))
