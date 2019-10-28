#!/bin/bash
echo -e "\nLoading global variables from 00-project-config.sh...\n"
source 00-project-config.sh

echo -e "project_id: $project_id"
echo -e "sa: $sa"

#echo -e "\nDeleting k8s-cluster-sa service account to be recreated again..."
#gcloud iam service-accounts delete $sa@$project_id.iam.gserviceaccount.com
#gcloud projects remove-iam-policy-binding $project_id --member serviceAccount:$sa@$project_id.iam.gserviceaccount.com --role "roles/owner"

echo -e "\nCreating a service account named $sa...\n"
gcloud iam service-accounts create $sa --display-name "$sa"

echo -e "\nAssigning owner role to the created service account ($sa)..."
echo -e "\nYou need to have permissions to change iam policies. Ask project owner to assign you the proper role...\n"
gcloud projects add-iam-policy-binding $project_id --member serviceAccount:$sa@$project_id.iam.gserviceaccount.com --role "roles/editor"

echo -e "\nCreating the key and storing it as service-key.json...\n"
gcloud iam service-accounts keys create service-key.json --iam-account $sa@$project_id.iam.gserviceaccount.com

echo -e "\nAuthorizing the service account...\n"
gcloud auth activate-service-account --key-file service-key.json

echo -e "\nCreating secret key...\n"
kubectl create secret generic histcnn-secret-key --from-file=key.json=service-key.json
# rm service-key.json

echo -e "\nCoping service-key.json to ~/.config/ ..."
cp service-key.json ~/.config/

echo -e "\nExporting the path..."
echo -e "\nFor the future use copy this path to your .bashrc or .bash_profile:"
credentials_path=`eval echo "~"`'/.config/service-key.json'
export GOOGLE_APPLICATION_CREDENTIALS="$credentials_path"
echo -e "\nexport GOOGLE_APPLICATION_CREDENTIALS=\"$GOOGLE_APPLICATION_CREDENTIALS\"\n"


echo -e "\nVerifying authentication..."
echo -e "\nShould return a list of existing buckets in the project:"
echo -e """
from google.cloud import storage
storage_client = storage.Client()
buckets = list(storage_client.list_buckets())
print(buckets)
""" > temp.py
python temp.py
rm temp.py
