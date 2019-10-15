from google.cloud import pubsub_v1 as pubsub
from google.cloud import datastore
from tqdm import tqdm
import numpy as np
import time
import os
from math import ceil

project_id = os.environ['project_id']
topic_name = os.environ['topic_name']
task_kind = os.environ['task_kind']
# saved_model_path = os.environ['saved_model_path']
# tfrecpath = os.environ['tfrecpath']
# image_file_metadata = os.environ['image_file_metadata']

publisher = pubsub.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

def create_client(project_id):
    return datastore.Client(project_id)
client = create_client(project_id)

def add_task(client, n, cancertype1, cancertype2):
    key = client.key(task_kind)

    task = datastore.Entity(
        key, exclude_from_indexes=[])

    task.update({
        'task_number': n,
        'status': 'Queued',
        'created': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#         'gcs_output_path': gcs_output_path,
#         'saved_model_path': saved_model_path,
#         'image_file_metadata': image_file_metadata,
#         'tfrecpath': tfrecpath,

        'cancertype1': cancertype1,
        'cancertype2': cancertype2,
        'completed_time' : None,
        'elapsed_time_s' : None,
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


import itertools
# cancertypes = ['blca', 'esca', 'kich', 'kirp', 'luad', 'ov',
#                'prad', 'sarc', 'thca', 'ucec','brca', 'coad', 'hnsc',
#                'kirc', 'lihc', 'lusc', 'paad', 'read', 'stad']
cancertypes = ['kirp', 'ucs', 'blca', 'esca', 'luad', 'tgct',
               'lusc', 'stad', 'brca', 'coadread', 'lihc']

combos = list(itertools.product(cancertypes, cancertypes))

n = -1
for cancertype1, cancertype2 in tqdm(combos):
    n+=1
    # Creating a task list in DataStore
    task = add_task(client, n, cancertype1=cancertype1, cancertype2=cancertype2)

    # Publishing the task.id to Pub/Sub
    data = str(task.id).encode('utf-8')
    publisher.publish(topic_path, data)

print('Published {:d} messages.'.format(n))
