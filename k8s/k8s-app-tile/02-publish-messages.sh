#!/bin/bash

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
echo -e "topic_name: $topic_name"
echo -e "subscription_name: $subscription_name"

echo -e "\nCopying latest histcnn from src directory..."
cp -rf ../../src/histcnn/ histcnn/

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
for cancertype_ in "${cancertypes[@]}"
do
  export cancertype=$cancertype_
  echo -e "\nPublishing DataStore and pub/sub messages for cancer type: $cancertype\n"
  python publish.py
done

echo -e "\nA sample message pulled (without acknowledgment):"
gcloud pubsub subscriptions pull $subscription_name
