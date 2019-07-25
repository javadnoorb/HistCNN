from google.cloud import pubsub_v1 as pubsub
from google.cloud import datastore
from tqdm import tqdm
from histcnn import (choose_input_list, util)
import numpy as np
import pandas as pd
import time
import os

project_id = os.environ['project_id']
topic_name = os.environ['topic_name']
task_kind = os.environ['task_kind']

publisher = pubsub.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

def create_client(project_id):
    return datastore.Client(project_id)

client = create_client(project_id)

def add_task(client, n, svs_path):
    key = client.key(task_kind)

    task = datastore.Entity(
        key, exclude_from_indexes=[])

    task.update({
        'task_number': n,
        'status': 'Queued',
        'created': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'bucket_name': None,
        'completed_time' : None,
        'elapsed_time_s' : None,
        'number_of_tiles' : None, # count_local_tiles
        'tiles_size_MBi' : None,
        'svs_path' : svs_path
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

svs_paths = pd.read_csv('/tmp/svs_path_list.txt', header=None).rename(columns={0:'svspath'})

n = 0
for svs_path in tqdm(svs_paths['svspath']):
    # Creating a task list in DataStore
    task = add_task(client, n, svs_path)

    # Publishing the task.id to Pub/Sub
    data = str(task.id).encode('utf-8')
    publisher.publish(topic_path, data)
    n += 1

print('Published {} messages.'.format(n))
