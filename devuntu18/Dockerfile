FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    # utils
    ca-certificates \
    less \
    vim \
    tmux \
    git \
    locales \
    language-pack-ja-base \
    language-pack-ja \
    # pyenv
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    && apt-get clean

RUN locale-gen ja_JP.UTF-8
ENV LANG='ja_JP.UTF-8'
COPY install_pyenv.sh /tmp/
WORKDIR /tmp
RUN sh install_pyenv.sh
RUN rm -f install_pyenv.sh
