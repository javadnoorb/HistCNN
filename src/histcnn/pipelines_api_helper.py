from pipelines.api import utils
from pipelines.api import Task, Workflow
from google.cloud import datastore
from google.cloud import storage


class datastore_tasker(object):
    def __init__(self, project_id):
        self.project_id = project_id
        self.gsclient = storage.Client(self.project_id)
        self.dsclient = datastore.Client(self.project_id)
        

    def create_task_database_in_datastore(self, cmd, project_id, tasknames, memory, cores, 
                                          disk_size, preemptible, description, docker_file, 
                                          local_outpath, output_path, logging, input_dict, tags):

        task1 = Task(tasknames[0]) ## task function takes any task name
        task1.set_description(description) # set description for a task
        task1.set_tags(tags=tags)
        task1.set_cores(cores=cores)
        task1.set_memory(memory=memory)

        task1.set_inputs(input_dict) 
        task1.set_output(output_path)
        task1.set_local_outpath(local_outpath)
        task1.set_disksize(disk_size=disk_size)
        task1.set_cmd((cmd))
        task1.set_preemptible(preemptible)
        task1.set_docker(docker_file)

        print(task1.to_json())

        # every task must be part of a workflow   
        workflow = Workflow(
                        name='nameme', 
                        project=project_id, ### <--- INSERT YOUR PROJECT ID HERE 
                        debug = False, 
                        logging = logging
                    )

        workflow.set_description(description)
        workflow.set_tags(tags)
        workflow.add_tasks(task1=task1)
        print(workflow.to_json())

        workflow.to_queue(taskgroup = tasknames[0], 
                          dsclient = self.dsclient, 
                          gsclient = self.gsclient)