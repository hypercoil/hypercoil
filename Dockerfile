FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
MAINTAINER Rastko Ciric
LABEL version="hypercoil-v0.1.z-prealpha"

# install python3-pip
RUN apt update && apt install python3-pip -y

RUN pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN useradd -m -s /bin/bash -G users hypercoil
WORKDIR /home/hypercoil
ENV HOME="/home/hypercoil"

# install software
RUN python -m pip install https://github.com/hypercoil/hypercoil/archive/main.zip

# testing and visualisation dependencies
RUN python -m pip install \
	pytest \
	https://github.com/wiheto/netplotbrain/archive/535a07bb7d2be1a2c95746bcb82b41e69f19a8f1.tar.gz \
	https://github.com/rciric/surfplot/archive/9b4a1e65aac89daf5e6747cee0419b6fc39d2392.tar.gz \
	scikit-image \
	pingouin \
	nilearn \
	setuptools \
	communities \
	brainspace

# entrypoint
CMD bash
