from google.cloud import pubsub_v1 as pubsub
from google.cloud import datastore
from tqdm import tqdm
import numpy as np
import time
import os
import sys
from math import ceil

project_id = os.environ['project_id']
topic_name = os.environ['topic_name']
task_kind = os.environ['task_kind']
category = os.environ['category']
cancertype = sys.argv[1]

tfrecords_output_bucket = os.environ['tfrecords_output_bucket']
shard_length = int(os.environ['shard_length'])
temp_path = os.environ['temp_path']

publisher = pubsub.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

def create_client(project_id):
    return datastore.Client(project_id)
client = create_client(project_id)

def add_task(client, n, cancertype, category, shard_length, gcs_output_path):
    key = client.key(task_kind)

    task = datastore.Entity(
        key, exclude_from_indexes=[])

    task.update({
        'task_number': n,
        'cancertype': cancertype,
        'status': 'Queued',
        'created': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'category' : category,
        'shard_length': shard_length,
        'shard_index': n,
        'gcs_output_path': gcs_output_path,
        'completed_time' : None,
        'elapsed_time_s' : None,
        'shard_length_tiles' : None,
        'tfrecord_size_MBi' : None
    })

    client.put(task)

    return task.key

def list_tasks(client):
    query = client.query(kind=task_kind)
    query.order = ['created']

    return list(query.fetch())

def delete_task(client, task_id):
    key = client.key('Task', task_id)
    client.delete(key)

image_file_metadata_filename = '{:s}/caches_basic_annotations.txt'.format(temp_path)
gcs_output_path = '{}/{}'.format(tfrecords_output_bucket, cancertype)

# Total number of tiles from image_file_metadata_filename
num_tiles = sum(1 for line in open(image_file_metadata_filename))-1
num_shards = ceil(num_tiles/shard_length)

for n in tqdm(range(num_shards)):
    # Creating a task list in DataStore
    task = add_task(client, n=n, cancertype=cancertype, category=category,
        shard_length=shard_length, gcs_output_path=gcs_output_path)

    # Publishing the task.id to Pub/Sub
    data = str(task.id).encode('utf-8')
    publisher.publish(topic_path, data)

print('Published {} messages.'.format(num_shards))
