FROM ubuntu:20.04

# https://github.com/anibali/docker-torch/blob/master/no-cuda/Dockerfile
# Use Tini as the init process with PID 1
ADD https://github.com/krallin/tini/releases/download/v0.10.0/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

RUN apt-get update 

# Install dependencies for OpenBLAS, and Torch
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git gfortran \
    # python3.8 as default on Ubuntu 20.04
    python3-numpy python3-nose python3-pandas \
    python3 python3-setuptools python3-dev \
    python3-h5py \
    pep8 python3-pip python3-wheel \
    python3-sphinx \
    # for python libs
    curl wget unzip libreadline-dev libjpeg-dev libpng-dev ncurses-dev \
    imagemagick gnuplot gnuplot-x11 libssl-dev libzmq3-dev graphviz \
    # OpenBLAS
    swig libopenblas-base \
    # HDF5
    libhdf5-dev \
    # cmake
    build-essential cmake libboost-system-dev libboost-thread-dev \
    libboost-program-options-dev libboost-test-dev libeigen3-dev \
    zlib1g-dev libbz2-dev liblzma-dev libboost-all-dev && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools 

# Clone this fork
WORKDIR /home/root/speech2text/vietasr
RUN git clone https://github.com/chukehill/vietasr.git
# Create simlink
RUN ln -s /usr/bin/python3.8 /usr/bin/python

WORKDIR /home/root/speech2text/vietasr/nemo/scripts/decoders/kenlm
RUN echo $(ls)
RUN mkdir build
WORKDIR /home/root/speech2text/vietasr/nemo/scripts/decoders/kenlm/build
RUN cmake ..
RUN make -j

WORKDIR /home/root/speech2text/vietasr/nemo/scripts/decoders
# python setup.py doesn't work properly in docker
RUN pip install .
# _swig_decoders missing error while importing ctc_decoders
#RUN cp build/lib.linux-x86_64-3.8/_swig_decoders.cpython-38-x86_64-linux-gnu.so .
RUN python ctc_decoders_test.py

WORKDIR /home/root/speech2text/vietasr
# seems like it works with v1.5.1, also it installs all needed packages
# but we still have to use the locally cloned repo in vietasr
RUN pip install https://github.com/kpu/kenlm/archive/master.zip flask transformers==4.9.2 soundfile datasets==1.11.0 pyctcdecode==v0.1.0
RUN pip install nemo_toolkit[all]==1.5.1
RUN python infer.py audio_samples
CMD ["python", "./app.py"]