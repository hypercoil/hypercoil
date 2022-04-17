FROM nvidia/cuda:11.6.2-cudnn8-devel-ubi8
MAINTAINER Rastko Ciric
LABEL version="hypercoil-v0.0.x-planning"

RUN useradd -m -s /bin/bash -G users hypercoil
WORKDIR /home/hypercoil
ENV HOME="/home/hypercoil"

# set up Python
RUN apt-get update -y
RUN apt-get install -y python3 --no-install-recommends
RUN python -m pip install --upgrade pip

# install software
RUN git clone https://github.com/rciric/hypercoil
RUN cd hypercoil
RUN git branch diffprog
RUN python -m pip install -e .

# testing and visualisation dependencies
RUN python -m pip install \
	pytest \
	https://github.com/wiheto/netplotbrain/archive/535a07bb7d2be1a2c95746bcb82b41e69f19a8f1.tar.gz \
	https://github.com/rciric/surfplot/archive/9b4a1e65aac89daf5e6747cee0419b6fc39d2392.tar.gz \
	scikit-image \
	pingouin \
	nilearn \
	setuptools \
	communities

# entrypoint
CMD bash
