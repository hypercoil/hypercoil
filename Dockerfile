FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
MAINTAINER Rastko Ciric
LABEL version="hypercoil-v0.0.x-planning"

RUN useradd -m -s /bin/bash -G users hypercoil
WORKDIR /home/hypercoil
ENV HOME="/home/hypercoil"

# install software
RUN python -m pip install https://github.com/rciric/hypercoil/archive/diffprog.zip

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
