#!/bin/bash

echo -e "\nRunning 02-publish-messages.sh to publish messages...\n"
yes | source 02-publish-messages.sh  # this is to bypass the question. To negate it: "yes n | source ..."

echo -e "\nRunning 03-build-push-docker.sh to build and push the docker image...\n"
source 03-build-push-docker.sh

echo -e "\nRunning 04-job-deployment.sh to deploy the app...\n"
source 04-job-deployment.sh

echo -e "\nDONE!"
