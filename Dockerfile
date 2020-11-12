FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    add-apt-repository multiverse


RUN apt-get -y update && \
    apt-get -y install  vim wget git && \
    pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*


ENV TERM xterm
ENV ZSH_THEME flazz

RUN apt-get -y update && \
    apt-get install -y zsh && \
    wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm requirements.txt

COPY . incremental_learning.pytorch/
RUN mkdir /checheckpoints/ /accuracy/
WORKDIR incremental_learning.pytorch