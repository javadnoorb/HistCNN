#!/bin/bash
echo -e "\nLoading global variables from ../00-project-config.sh...\n"
source ../00-project-config.sh

echo -e "\nLoading app variables from 00-app-config.sh...\n"
source 00-app-config.sh

echo -e "project_id: $project_id"
echo -e "deployment_name: $deployment_name"
echo -e "max_replica: $max_replica"
echo -e "image_name: $image_name"

echo -e "\nGet authentication credentials for the cluster..."
gcloud container clusters get-credentials --internal-ip $cluster_name

echo -e "\nConfirm cluster is running"
gcloud container clusters list

echo -e "\nConfirm connection to cluster"
kubectl cluster-info

echo -e """
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: $deployment_name
spec:
  # Pod specification
  replicas: $max_replica
  selector:
    matchLabels:
      run: $deployment_name
  template:
    metadata:
      creationTimestamp: null
      labels:
        run: $deployment_name
    spec:
      containers:
      - image: gcr.io/$project_id/${image_name}:v1
        imagePullPolicy: Always
        name: $deployment_name
        resources:
          requests:
            cpu: 2000m
            memory: 5000Mi
        volumeMounts:
        - name: google-cloud-key
          mountPath: /var/secrets/google
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /var/secrets/google/key.json      
      restartPolicy: Always          
""" > deployment.yaml

kubectl apply -f deployment.yaml

rm deployment.yaml

sleep 10s

#echo -e "\nEnabling horizontal pod autoscaling (HPA)..."
#kubectl autoscale deployment $deployment_name --min=1 --max=$max_replica --cpu-percent=20

echo -e "\nGet a list of running pods."
kubectl get pods
