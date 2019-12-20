FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

# Linuxパッケージのインストール
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && apt-get install -y \
    git \
    autoconf \
    tmux \
    vim \
    wget \
    cmake \
    byobu \
    language-pack-ja \
    unzip \
    nscd \
    graphviz \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    protobuf-compiler \
    python-tk \
    python-pil \
    python-lxml \
    python-opengl \
    xvfb \
    ffmpeg  \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*
	
# 言語設定
RUN locale-gen en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
RUN update-locale LANG=$LANG

# Pythonライブラリのインストール
RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
COPY . /tmp/

# pybullet-gymのインストール
RUN git clone https://github.com/benelot/pybullet-gym
RUN cd pybullet-gym \
&& git checkout 55eaa0defca7f4ae382963885a334c952133829d \
&& pip install -e .

# Tensorboardのポート番号
EXPOSE 6006

# Jupyter Notebookのポート番号
EXPOSE 8888

# WS Server
EXPOSE 5005