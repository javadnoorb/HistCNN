# Kubernetes Cluster Setup
This folder contains the scripts to launch a Kubernetes (k8s) cluster. Subsequent folders contain apps that can be launched on the cluster. To create the cluster itself and necessary dependancies execute the following scripts in the right order:

1. **01-set-or-create-k8s-cluster.sh:** When this script is run for the first time, it will create a config file that contains custom variable names locally on a file called 00-project-config.sh. This file won't be pushed to the repo. This file will be loaded during the app and cluster launches to get the appropriate config parameters. It will then perform the following tasks:
  - Sets the default project-id, region, and zone.
  - Updates gcloud components.
  - Enables Kubernetes API.
  - Creates a cluster on the autoscaling mode. The cluster specifications should be set in the script before the script is run. Certain parameters can be modified after the cluster is launched, but not all.
  - Note: If the cluster already exists it will set it locally, which is needed for running the apps.
  - The cluster is then authenticated and can be accessed with the `kubectl` CLI.
  - note: once the cluster is set up, it can be used for launching different apps. Since it is on the autoscaling mode it should be able to automatically be scaled down to 1-3 nodes. If you won't need the cluster in the upcoming days you can terminate it from the console UI, otherwise can be left on for the future tasks.
2. **02-set-service-account-once-per-project.sh:** This script will create a service account to be used for the machine that is running this script as well as for the docker container to communicate with goolge.cloud. The corresponding key will be saved in the same directory. The script will also make a copy of this key to /.config/service-key.json. One should add the `export` command to the .bashrc or .bash_profile. This script is not needed to be run more than once per project per laptop. The key saved in the k8s folder will be used for launching apps.

Note: Any of the shell scripts can be executed by "`. <SCRIPT-NAME>`" (or "`source <SCRIPT-NAME>`")
