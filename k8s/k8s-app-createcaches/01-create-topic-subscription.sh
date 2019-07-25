#!/bin/bash

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
  echo -e "export bucket_name='YOUR-BUCKET-NAME'  # Bucket to store the tiles\n" >> 00-app-config.sh

  echo -e "export cancertypes=('acc' 'blca' 'brca' 'cesc' 'chol' 'coad' 'dlbc' 'esca' 'gbm' 'hnsc' 'kich' 'kirc' 'kirp' 'lgg' 'lihc' 'luad' 'lusc' 'meso' 'ov' 'paad' 'pcpg' 'prad' 'read' 'sarc' 'skcm' 'stad' 'tgct' 'thca' 'thym' 'ucec' 'ucs' 'uvm')  # Cancer types" >> 00-app-config.sh  
  echo -e "export task_kind='Task:caching'  # Task type for DataStore" >> 00-app-config.sh
  echo -e "export topic_name='caching'  # Topic for Pub/Sub" >> 00-app-config.sh
  echo -e "export tiles_input_path='tcga_tiles'  # Input path of tiles on GCS" >> 00-app-config.sh
  echo -e "export subscription_name='caching_sub'  # Subscription for Pub/Sub" >> 00-app-config.sh
  echo -e "export image_name='caching'  # Image name used for docker and Container Registry" >> 00-app-config.sh
  echo -e "export deployment_name='caching'  # will be used for deployment, container, and pod." >> 00-app-config.sh
  echo -e "export max_replica=35  # will be used for the max replica to be launched in deplyoment" >> 00-app-config.sh
  echo -e "export ack_deadline_seconds=60  # Pub/Sub deadline from 0 to 600" >> 00-app-config.sh

  return
fi

echo -e "\nLoading app variables from 00-app-config.sh...\n"
source 00-app-config.sh

credentials_path=`eval echo "~"`'/.config/service-key.json'
export GOOGLE_APPLICATION_CREDENTIALS="$credentials_path"

echo -e "project_id: $project_id"
echo -e "zone_name: $zone_name"
echo -e "cluster_name: $cluster_name"
echo -e "tiles_input_path: $tiles_input_path"
echo -e "topic_name: $topic_name"
echo -e "subscription_name: $subscription_name"

echo -e "\nSetting up topic and subscription:"
python pubsub-setup.py

echo -e "\nListing all existing subscriptions:"
gcloud pubsub subscriptions list
