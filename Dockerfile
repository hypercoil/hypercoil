FROM nvidia/cuda:11.6.2-cudnn8-devel-ubi8
MAINTAINER Rastko Ciric
LABEL version="hypercoil-v0.0.x-planning"

RUN useradd -m -s /bin/bash -G users hypercoil
WORKDIR /home/hypercoil
ENV HOME="/home/hypercoil"

# set up Python
RUN yum update -y
RUN yum install -y python3
RUN python3 -m pip install --upgrade pip

# entrypoint
CMD bash
