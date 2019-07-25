from google.cloud import pubsub_v1 as pubsub
from google.cloud import datastore
from tqdm import tqdm
from histcnn import (choose_input_list, util)
import numpy as np
import time
import os

project_id = os.environ['project_id']
topic_name = os.environ['topic_name']
tile_size = int(os.environ['tile_size'])
cancertype = os.environ['cancertype']
task_kind = os.environ['task_kind']

publisher = pubsub.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

def create_client(project_id):
    return datastore.Client(project_id)
client = create_client(project_id)

def add_task(client, n, cancertype, multislide_size, tile_size,
            output_path, image_file_metadata_filename, gcs_output_path):
    key = client.key(task_kind)

    task = datastore.Entity(
        key, exclude_from_indexes=[])

    task.update({
        'task_number': n,
        'cancertype': cancertype,
        'status': 'Queued',
        'created': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'multislide_index': n,
        'multislide_size': multislide_size,
        'tile_size': tile_size,
        'output_path': output_path,
        'image_file_metadata_filename': image_file_metadata_filename,
        'gcs_output_path': gcs_output_path,
        'bucket_name': None,
        'completed_time' : None,
        'elapsed_time_s' : None,
        'number_of_tiles' : None, # count_local_tiles
        'tiles_size_MBi' : None,
        'SVS_filename' : None
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

svs_metadata  = choose_input_list.get_svs_metadata(cancertype)
svs_metadata.to_csv('data/svs_metadata_{:s}.txt'.format(cancertype))

main_tiles_subdirecory = '{can}_{size}x{size}'.format(can=cancertype, size=tile_size)
output_path = '/mnt/data/output/'
gcs_output_path = 'tcga_tiles/{:s}/{:s}/'.format(cancertype, main_tiles_subdirecory)
image_file_metadata_filename = 'data/svs_metadata_{:s}.txt'.format(cancertype)
multislide_size = 1
num_slides = util.count_lines(image_file_metadata_filename) - 1
num_tasks = int(np.ceil(num_slides/multislide_size))

for n in tqdm(range(num_tasks)):
    # Creating a task list in DataStore
    task = add_task(client, n=n, cancertype=cancertype, multislide_size=multislide_size,
                tile_size=tile_size, output_path=output_path,
                image_file_metadata_filename=image_file_metadata_filename,
                gcs_output_path=gcs_output_path)

    # Publishing the task.id to Pub/Sub
    data = str(task.id).encode('utf-8')
    publisher.publish(topic_path, data)

print('Published {} messages.'.format(num_tasks))
