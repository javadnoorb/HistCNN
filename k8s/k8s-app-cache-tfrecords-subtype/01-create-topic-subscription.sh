#!/bin/bash

tissue="kidney"

echo -e "\nThis script assumes the cluster $cluster_name in project $project_id is up and running."
echo -e "If it's not the case please first create a cluster using ../01-create-k8s-cluster.sh"
echo -e "\nThis script also assumes you have created the service key ../service-key.json"
echo -e "If this is not true please first run ../02-set-service-account-once-per-project.sh"

echo -e "\nLoading global variables from ../00-project-config.sh...\n"
source ../00-project-config.sh

if [ ! -e 00-app-config.sh ]; then
  echo -e "\nError: 00-app-config.sh not found!"
  echo -e "A template was created for you. Please complete it with your specific app values and try again.\n"
  echo "#!/bin/bash" > 00-app-config.sh
  echo -e "export tissue='$tissue' # tissue type" >> 00-app-config.sh
  echo -e "export caches_input_bucket='histology-tmp'  # Bucket name will be used to get the tiles from" >> 00-app-config.sh
  echo -e "export tfrecords_output_bucket=gs://histology-cnn/tfrecords/cache_subtypes/\$tissue  # Bucket name and path to store tfrecords to\n" >> 00-app-config.sh
  job_tag="tfrecords_cache_subtype"
  echo -e "export task_kind='Task:$job_tag'  # Task type for DataStore" >> 00-app-config.sh
  echo -e "export topic_name='create_$job_tag'  # Topic for Pub/Sub" >> 00-app-config.sh
  echo -e "export subscription_name='create_${job_tag}_sub'  # Subscription for Pub/Sub" >> 00-app-config.sh
  echo -e "export image_name='create_$job_tag'  # Image name used for docker and Container Registry" >> 00-app-config.sh
  echo -e "export category='all'  # Creates tfrecords for each category" >> 00-app-config.sh
  echo -e "export shard_length=10000  # length of shards" >> 00-app-config.sh
  echo -e "export deployment_name='${job_tag//_/-}'  # will be used for deployment, container, and pod." >> 00-app-config.sh
  echo -e "export max_replica=1000  # will be used for the max replica to be launched in deplyoment" >> 00-app-config.sh
  echo -e "export ack_deadline_seconds=60  # Pub/Sub deadline from 0 to 600" >> 00-app-config.sh
  echo -e "export temp_path='$(mktemp -d --suffix .histcnn.subtype)' # a temporary path to be used for different purposes" >> 00-app-config.sh
  echo -e "export annfile=subtype_ann_\${tissue}_tumor.txt # annotations file name" >> 00-app-config.sh
return
fi

source 00-app-config.sh
echo -e "\nSetting up topic and subscription:"
python pubsub-setup.py

echo -e "\nListing all existing subscriptions:"
gcloud pubsub subscriptions list
