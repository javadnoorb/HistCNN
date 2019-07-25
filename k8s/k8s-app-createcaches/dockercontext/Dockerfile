FROM ubuntu:16.04

RUN apt-get update && apt-get install -y gcc wget bzip2 openslide-tools vim
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda update conda
RUN conda install python=3.6  # tensorflow doesn't work with 3.7
RUN conda install -c anaconda pip
RUN conda create -n py2 python=2.7 -y
# Downloading gcloud package
RUN wget https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -O /tmp/google-cloud-sdk.tar.gz

# install python packages
RUN pip install --upgrade pip && pip install \
    openslide-python \
    flask \
    pandas-gbq \
    tensorflow \
    google-cloud \
    google-cloud-storage \
    google-cloud-pubsub \
    google-cloud-datastore \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    scikit-image \
    tqdm

# Installing gcloud-sdk
RUN mkdir -p /usr/local/gcloud
RUN  tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz
RUN /usr/local/gcloud/google-cloud-sdk/install.sh --quiet
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
ENV CLOUDSDK_PYTHON /root/miniconda3/envs/py2/bin/python

WORKDIR /app

# Adding credentials for the test container
COPY dock_worker.py service-key.json ./
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-key.json

# Install histcnn
COPY HistCNN/ ./HistCNN/
RUN mkdir ./data
RUN cd HistCNN && pip install --upgrade pip && pip install . 
RUN rm -r HistCNN

# run the worker code
CMD python dock_worker.py
