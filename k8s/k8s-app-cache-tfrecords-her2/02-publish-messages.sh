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
echo -e "cancertypes: ${cancertypes[*]}"
echo -e "gcs_ann_path: $gcs_ann_path"
echo -e "temp_path: $temp_path"

clear_subscription()
{
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
}

publish_jobs_single_cancertype()
{
    cancertype=$1
    echo -e "Saving temporary files to $temp_path"
    caches_path="gs://${caches_input_bucket}/tcga_tiles/pancancer/$cancertype/${cancertype}_512x512_cache/"

    histcnn gcs --checkpath "$gcs_ann_path/$cancertype/caches_gcs_path_list.txt" --download_dir $temp_path --project $project_id

    if [ $? -ne 0 ]; then
      echo -e "Fetching tile names on GCS ($caches_path). This may take several minutes..."
      gsutil -m ls "${caches_path}*.svs/*.txt" > "$temp_path/caches_gcs_path_list.txt"
      gsutil cp "$temp_path/caches_gcs_path_list.txt" "$gcs_ann_path/$cancertype/caches_gcs_path_list.txt"
      echo -e "Saved to $gcs_ann_path/$cancertype/caches_gcs_path_list.txt"
    fi

    echo -e "Printing a few lines from the file:"
    head -n 5 $temp_path/caches_gcs_path_list.txt

    histcnn gcs --checkpath "$gcs_ann_path/$cancertype/caches_basic_annotations.txt" --download_dir $temp_path --project $project_id
    if [ $? -ne 0 ]; then
      echo -e "\nExecuting assign_validation_and_other_labels_to_caches.py:"
      python assign_validation_and_other_labels_to_caches.py $temp_path
      gsutil cp "$temp_path/caches_basic_annotations.txt" "$gcs_ann_path/$cancertype/caches_basic_annotation.txt"
      echo -e "Saved to $gcs_ann_path/$cancertype/caches_basic_annotation.txt"
    fi

    # Publishing messages
    echo -e "\nPublishing the messages:"
    python publish.py
}

clear_subscription
for cancertype in "${cancertypes[@]}"
do
  echo -e "\nFetching metadata, and publishing DataStore and pub/sub messages for cancer type: $cancertype\n"
  publish_jobs_single_cancertype $cancertype
done


echo -e "\nA sample message pulled (without acknowledgment):"
gcloud pubsub subscriptions pull $subscription_name
