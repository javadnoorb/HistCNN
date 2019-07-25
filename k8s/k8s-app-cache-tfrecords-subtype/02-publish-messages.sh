#!/bin/bash

echo -e "\nThis script creates metadata for tfrecords and publishes shard number for the k8s app."

echo -e "\nThis script assumes the cluster $cluster_name in project $project_id is up and running."
echo -e "If it's not the case please first create a cluster using ../01-create-k8s-cluster.sh"
echo -e "\nThis script also assumes you have created the service key ../service-key.json"
echo -e "If this is not true please first run ../02-set-service-account-once-per-project.sh"

echo -e "\nLoading global variables from ../00-project-config.sh...\n"
source ../00-project-config.sh

echo -e "\nLoading app variables from 00-app-config.sh...\n"
source 00-app-config.sh

credentials_path=`eval echo "~"`'/.config/service-key.json'
export GOOGLE_APPLICATION_CREDENTIALS="$credentials_path"

echo -e "project_id: $project_id"
echo -e "zone_name: $zone_name"
echo -e "cluster_name: $cluster_name"
echo -e "caches_input_bucket: $caches_input_bucket"
echo -e "topic_name: $topic_name"
echo -e "subscription_name: $subscription_name"
echo -e "temp_path: $temp_path"

# echo -e "Saving temporary files to $temp_path"
# gcs_path="gs://histology-cnn/data/pancancer_annotations/pancancer"

download_annfile=0
if [ -f $temp_path/$annfile ]; then
 read -p "Annotation file $annfile already exists in $temp_path. Would you like to build and rewrite it again? (y/n) " -n 1 -r
 echo
 if [[ $REPLY =~ ^[Yy]$ ]]; then download_annfile=1; fi
else download_annfile=1
fi

if [ $download_annfile -eq 1 ]; then
    histcnn annotate --gcs-path-list /sdata/data/pancancer_annotations/individual_cancers/  --output-file \
    $temp_path/$annfile --drop-normals --lstrip-string gs://histology-tmp/ --tissue $tissue
#   histcnn gcs --project $project_id --checkpath "$gcs_path/$annfile" --download_dir $temp_path
#   if [ $? -ne 0 ]; then
#     echo -e "$gcs_path/$annfile does not exist. Please add it to GCS"
#     return
#   fi
#   echo -e 'Download complete.'
fi



echo -e "Printing a few lines from the file:"
head -n 5 $temp_path/$annfile

# Publishing messages
read -p "Would you like to clear up the queue from the subscription $subscription_name? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo -e "\nDeleting the old subscription..."
  gcloud pubsub subscriptions delete $subscription_name

  sleep 10s

  echo -e "\nCreating a pull subscription..."
  gcloud pubsub subscriptions create $subscription_name --ack-deadline=600 --topic=$topic_name --topic-project=$project_id

  echo -e "\nChecking the queue is empty..."
  gcloud pubsub subscriptions pull $subscription_name
fi

echo -e "\nPublishing the messages:"
python publish.py

echo -e "\nA sample message pulled (without acknowledgment):"
gcloud pubsub subscriptions pull $subscription_name
