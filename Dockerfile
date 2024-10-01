ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/ASVLeipzig/cor-asv-ann/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/ASVLeipzig/cor-asv-ann" \
    org.label-schema.build-date=$BUILD_DATE

SHELL ["/bin/bash", "-c"]
WORKDIR /build/cor-asv-ann
RUN pip install nvidia-pyindex && \
    pushd $(mktemp -d) && \
    pip download --no-deps "nvidia-tensorflow==1.15.5+nv22.12" && \
    for name in nvidia_tensorflow-*.whl; do name=${name%.whl}; done && \
        python -m wheel unpack $name.whl && \
        for name in nvidia_tensorflow-*/; do name=${name%/}; done && \
        newname=${name/nvidia_tensorflow/tensorflow_gpu} &&\
        sed -i s/nvidia_tensorflow/tensorflow_gpu/g $name/$name.dist-info/METADATA && \
        sed -i s/nvidia_tensorflow/tensorflow_gpu/g $name/$name.dist-info/RECORD && \
        sed -i s/nvidia_tensorflow/tensorflow_gpu/g $name/tensorflow_core/tools/pip_package/setup.py && \
        pushd $name && for path in $name*; do mv $path ${path/$name/$newname}; done && popd && \
        python -m wheel pack $name && \
        pip install $newname*.whl && popd && rm -fr $OLDPWD
# - preempt conflict over numpy between scikit-image and tensorflow
# - preempt conflict over numpy between tifffile and tensorflow (and allow py36)
RUN pip install imageio==2.14.1 "tifffile<2022"
# - preempt conflict over numpy between h5py and tensorflow
RUN pip install "numpy<1.24"

COPY setup.py .
COPY ocrd-tool.json .
COPY ocrd_cor_asv_ann ./ocrd_cor_asv_ann
COPY requirements.txt .
COPY README.md .
RUN pip install .
RUN rm -fr /build/cor-asv-ann

WORKDIR /data
VOLUME ["/data"]
