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
                     #choose_input_list,
                     handle_google_cloud_apis)
import os
import sys

project_id = PROJECT_ID
subscription_name = SUBSCRIPTION_NAME
bucket_name = BUCKET_NAME
task_kind = TASK_KIND

subscriber = pubsub.SubscriberClient()
subscription_path = subscriber.subscription_path(
    project_id, subscription_name)

NUM_MESSAGES = 1
ACK_DEADLINE = 60
SLEEP_TIME = 30

def mark_done(client, task_id, completed_time, elapsed_time_s,
              number_of_tiles, tiles_size_MBi, SVS_filename):
    with client.transaction():
        key = client.key(task_kind, task_id)
        task = client.get(key)

        if not task:
            raise ValueError(
                '{} {} does not exist.'.format(task_kind, task_id))

        task['status'] = 'Done'
        task['completed_time'] = completed_time
        task['elapsed_time_s'] = elapsed_time_s
        task['number_of_tiles'] = number_of_tiles
        task['tiles_size_MBi'] = tiles_size_MBi
        task['SVS_filename'] = SVS_filename
        task['bucket_name'] = bucket_name
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

    image_file_metadata_filename = params['image_file_metadata_filename']
    multislide_index = int(params['multislide_index'])
    tile_size = int(params['tile_size'])
    output_path = params['output_path']
    gcs_output_path = params['gcs_output_path']
    multi_slide_size = int(params['multislide_size'])

    df = pd.read_csv(image_file_metadata_filename, index_col=0)

    for slide_index in range(multislide_index*multi_slide_size,
                             min((multislide_index+1)*multi_slide_size, len(df))):

        slide_info = df.iloc[slide_index]

        if slide_info['AppMag'] == 20:
            downsample = 1
        elif slide_info['AppMag'] == 40:
            downsample = 2
        else:
            raise Exception('Unknown optical zoom.')

        process_files.DownloadSVS_TileIt_DeleteIt(slide_info, output_path,
                                                  downsample = downsample,
                                                  tile_size = tile_size, bg_th = 220,
                                                  max_bg_frac = 0.5,
                                                  project=project_id)

        bucket = handle_google_cloud_apis.gcsbucket(project_name=project_id, bucket_name=bucket_name)
        bucket.copy_files_to_gcs(output_path, gcs_output_path, verbose=True)

        # Counting local tiles
        local_files = output_path + slide_info['svsFilename'] + '/tiles/*.jpg'
        command = "ls -l " + local_files + " | wc -l"
        count_local_tiles_ = os.popen(command).read()
        count_local_tiles = int(count_local_tiles_.split()[0])

        # size of Tiles
        command = 'du -s ' + output_path + slide_info['svsFilename'] + '/'
        tiles_size = round(int(os.popen(command).read().split()[0])/1000,1)  # in MB

        # Counting gcs tiles
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(bucket_name)
        directory_name = gcs_output_path + slide_info['svsFilename'] + '/tiles/'
        blobs = bucket.list_blobs(prefix=directory_name)
        count_gcs_tiles = 0
        for blob in blobs:
            count_gcs_tiles += 1

        # Removing local files
        command = "rm -rf " + output_path
        os.popen(command)

        elapsed_time_s = round((time.time() - start_time), 1)  # in seconds

        completed_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # We will comfirm the job once we made sure the number of created tiles matches with the ones copied to gcs...
        if count_local_tiles == count_gcs_tiles:

            client = datastore.Client(project_id)
            mark_done(client=client, task_id=task_id, completed_time=completed_time,
                      elapsed_time_s=elapsed_time_s, number_of_tiles=count_local_tiles,
                      tiles_size_MBi=tiles_size, SVS_filename=slide_info['svsFilename'])

            print('Completed tiling for SVS file: {}'.format(slide_info['svsFilename']))
            print('Created {} tiles, total size of {} MB and copied to: gs://{}/{}'.format(count_local_tiles, tiles_size, bucket_name, directory_name))
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
