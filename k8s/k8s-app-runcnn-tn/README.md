# Create tfrecords (k8s App)
This folder contains a Kubernetes app that runs the Inception-V3 CNN from the tfrecords located on a GCS bucket. 

## Prerequisites
* Make sure a cluster with `cluster_name` set in ../00-project-config.sh is up and running. If not follow the steps in the parent folder of this app folder to create one.
* DataStore should be enabled for the current project. Recommended mode is FireStore in DataStore mode.
* Tiles should be ready for use and on a GCS bucket.

## Steps
In order to launch this app execute the following scripts in the following order:

1. **01-create-topic-subscription.sh:** When this script is run for the first time, it will create a config file that contains custom variable names locally on a file called 00-app-config.sh. This file won't be pushed to the repo. This file will be loaded during the app launches to get the appropriate config parameters. Then, this script will remove the existing Cloud Pub/Sub topic and subscription and create a topic with a subscription. The names are set in 00-app-config.sh. It only removes the `topic/subscription` with the names defined in the config file. This script can be run only once per project and reused for various `create_tfrecords_runcnn_tn` jobs.
2. **02-publish-messages.sh:** This scripts creates a queue for the cancertypes to run the CNN on. This queue is first published to Cloud DataStore, and the `task_id` will be published through Pub/Sub and will be pulled by the running app in k8s. You are given the option to empty the queue before publishing new messages. You can skip clearing the queue if you would like to add new tasks to the existing ones.
3. **03-build-push-docker.sh:** This script will create a docker and push it to Google Registry. It copies all the necessary files such as most recent histcnn folder and service-key.json. It will prepare the worker script named dock_worker.py to be copied to the container. This python script has the appropriate parameters, such as `bucket_name`, `project_id`, `task_kind`, and `subscription_name`. Follow the printed messages to test the container locally. The name of the image: `gcr.io/$project_id/create_tfrecords_runcnn_tn`
4. **04-job-deployment.sh:** This script will deploy a deployment on our k8s cluster that we've made. The most important parameter in this script is `max_replica`, which is the number of pods that will be running our application.
5. **137-run-pancancer.sh:** This script will run steps 02, 03, and 04 in one step. As mentioned above step 01 is only needed once for each project, unless the pub/sub topic was somehow removed.

**worker.py**: is responsible for running the process. It will pull a message (`task_id`) from Pub/Sub, then using that obtain all the necessary parameters from DataStore. It will update the `status` flag to `InProgress` while work starts and `Done` when it successfully finishes. As of running this README file there is no check for "success". It will output a few metrics to DataStore, these include number of tiles in the create_tfrecords job, size in MBi, completed time, and elapsed time. All the local files are deleted after the upload to GCS and these measurements.
