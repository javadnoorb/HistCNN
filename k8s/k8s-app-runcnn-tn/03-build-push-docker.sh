#!/bin/bash
echo -e "\nLoading global variables from ../00-project-config.sh...\n"
source ../00-project-config.sh

echo -e "Loading app variables from 00-app-config.sh...\n"
source 00-app-config.sh

echo -e "project_id: $project_id"
echo -e "zone_name: $zone_name"
echo -e "cluster_name: $cluster_name"
echo -e "image_name: $image_name"

echo -e "\nCopying ../service-key.json to docker context"
cp ../service-key.json dockercontext/

echo -e "\nCopying histcnn files"
histcnn_root_dir=$(git rev-parse --show-toplevel)
mkdir -p dockercontext/HistCNN
cp -r $histcnn_root_dir/setup.py dockercontext/HistCNN/
cp -r $histcnn_root_dir/REQUIREMENTS.txt dockercontext/HistCNN/
cp -r $histcnn_root_dir/src dockercontext/HistCNN/

echo -e "\nCreating a temp file for docker to replace the appropriate project values:"
cp worker.py dockercontext/dock_worker.py
sed -i -e 's/PROJECT_ID/"'$project_id'"/g' dockercontext/dock_worker.py
sed -i -e 's/SUBSCRIPTION_NAME/"'$subscription_name'"/g' dockercontext/dock_worker.py
sed -i -e 's/INPUT_BUCKET/"'$input_bucket'"/g' dockercontext/dock_worker.py
sed -i -e 's/TASK_KIND/"'$task_kind'"/g' dockercontext/dock_worker.py
sed -i -e 's|ANNOTATIONS_PATH|"'$annotations_path'"|g' dockercontext/dock_worker.py
sed -i -e 's|RESULTS_PATH|"'$results_path'"|g' dockercontext/dock_worker.py
sed -i -e 's|PANCANCER_TFRECORDS_PATH|"'$pancancer_tfrecords_path'"|g' dockercontext/dock_worker.py

echo -e "\nBuilding a docker image called ${image_name}:"

echo -e '\nDocker context contents:'



# Note that in the following code ssh private key will not leave a trace in the image
# This is due to the multi-stage implementation of dockerfile
docker build --build-arg project_id=$project_id \
			 -t gcr.io/$project_id/${image_name}:v1 \
             ./dockercontext/

echo -e "\nPushing it to Container Registry:"
docker push gcr.io/$project_id/${image_name}:v1

echo -e "\nRemoving the temp files..."
rm dockercontext/dock_worker*
rm dockercontext/service-key.json
rm -r dockercontext/HistCNN/

echo -e "\nTo interact with this container locally execute the following:"
echo -e "docker run -it gcr.io/$project_id/${image_name}:v1 bash\n"

echo -e "\nTo test if container is running execute the following:"
echo -e "docker run -t gcr.io/$project_id/${image_name}:v1\n"
