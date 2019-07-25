#!/bin/bash
if [ ! -e 00-project-config.sh ]; then
  echo -e "Error: 00-project-config.sh not found! A template was created for you. Please complete it with your specific project/cluster values and try again.\n"
  echo "#!/bin/bash" > 00-project-config.sh
  echo -e "export project_id='jax-nihcc-res-00-0011' # Fill me out with proper project name" >> 00-project-config.sh
  echo -e "export sa='k8s-sa-xxx' # Name for service account - fill xxx with your initials\n" >> 00-project-config.sh
  echo -e "export cluster_name='k8s-cluster'" >> 00-project-config.sh
  echo -e "export region_name='us-east1'" >> 00-project-config.sh
  echo -e "export zone_name='us-east1-b'" >> 00-project-config.sh
  echo -e "export max_node=8" >> 00-project-config.sh
  return
fi

echo -e "\nLoading global variables from 00-project-config.sh...\n"
source 00-project-config.sh

echo -e "project_id: $project_id"
echo -e "zone_name: $zone_name"
echo -e "cluster_name: $cluster_name"
echo -e "max_node: $max_node"

if [ ! `gcloud config get-value core/project 2> /dev/null` = $project_id ]; then
  echo -e "Error: The project_id that was set in 00-project-config.sh ($project_id) doesn't match with your default GCP project_id."
  echo "Use the following command to set it as default:"
  echo -e "gcloud config set project $project_id\n"
  return
fi

echo -e "Default region was set to `gcloud config get-value compute/region 2> /dev/null`."
echo -e "Setting it to $region_name. In order to change it your preferred region try this:"
echo -e "gcloud config set compute/region <REGION>\n"
gcloud config set compute/region $region_name

echo -e "Default zone was set to `gcloud config get-value compute/zone 2> /dev/null`."
echo -e "Setting it to $zone_name. In order to change it your preferred zone try this:"
echo -e "gcloud config set compute/zone <ZONE>\n"
gcloud config set compute/zone $zone_name

echo -e "Update gcloud to the latest version"
gcloud components update

echo -e "\nEnabling Kubernetes API"
gcloud services enable container.googleapis.com

echo -e "\nInstalling/updating kubectl:"
gcloud components install kubectl

echo -e "\nCreating a Kubernetes cluster named $cluster_name\n"
set -o xtrace
gcloud container clusters create $cluster_name \
  --machine-type=n1-highmem-8 \
  --num-nodes=1 --preemptible \
  --disk-type=pd-ssd --disk-size=200GB \
  --enable-autoscaling \
  --min-nodes=0 --max-nodes=$max_node \
  --zone $zone_name \
  --scopes storage-rw
set +o xtrace

echo -e "\nIf you got quota error either lower the number of CPUs in the query or consider increasing your cpu limit to a higher number by following these steps:"
echo -e "https://cloud.google.com/compute/quotas"
echo -e "If you are using a free tier you might need to upgrade it."

echo -e "\nGet authentication credentials for the cluster..."
gcloud container clusters get-credentials $cluster_name

echo -e "\nConfirm cluster is running"
gcloud container clusters list

echo -e "\nConfirm connection to cluster"
kubectl cluster-info

echo -e "\nGet a list of running pods."
kubectl get pods

echo -e "\nCluster $cluster_name was successfully created!"
