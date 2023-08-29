FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG USERNAME
ARG USER_ID
ARG GROUP_ID

#RUN apt install sudo=1.8.31-1ubuntu1 -y
RUN groupadd --gid ${GROUP_ID} $USERNAME
RUN useradd --uid ${USER_ID} --gid ${GROUP_ID} --shell /bin/bash --create-home --no-user-group ${USERNAME}
RUN chown ${USER_ID}:${GROUP_ID} /mnt/

# install usefull linux packages
RUN apt update \
&& apt install git -y \
&& apt install zip -y \
&& apt install unzip -y \
&& apt install wget -y \
&& apt install curl -y \
&& apt install screen -y


ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"
ARG MMCV="2.0.1"


RUN  apt-get update && apt-get install -y gnupg \
&&  apt-get clean && rm -rf /tmp/* /var/tmp/*


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all



# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"

USER ${USER_ID} 


# Install MMSegmentation
RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git  /mnt/mmseg
WORKDIR  /mnt/mmseg
ENV FORCE_CUDA="1"
# RUN pip freeze > requirements.txt
RUN pip install  -r requirements.txt
RUN pip install --no-cache-dir -v -e .



# install usefull python packages
RUN pip install --upgrade pip \
&& pip install ipython \
&& pip install ipykernel \
&& pip install pandas==2.0.0 \
&& pip install matplotlib==3.7.1 \
&& pip install -U scikit-learn==1.2.2 \
#&& pip install tensorboard \
&& pip install PyYAML==6.0

RUN pip install torcheval==0.0.6 \
&& pip install torchsummary==1.5.1 \
&& pip install torchmetrics==0.11.4 \
&& pip install lightning==2.0.2 \
&& pip install -U tensorboard-plugin-profile==2.11.2
  

RUN git config --global user.name ${USERNAME} \
&& git config --global user.email kyriakifana@gmail.com


USER root
RUN chown -R ${USER_ID}:${GROUP_ID} /mnt/
