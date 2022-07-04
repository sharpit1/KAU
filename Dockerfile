FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
MAINTAINER sharpit
RUN apt-get -y update
RUN apt update
RUN apt-get -y install python3.7
RUN apt-get -y install python3-pip
RUN apt-get -y install git
RUN apt-get -y install zlib1g-dev
RUN apt-get -y install libjpeg-dev
RUN apt-get -y install wget
RUN apt-get -y install curl
RUN sh -c 'curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg'
RUN sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
RUN apt update
RUN apt-get -y install code
RUN pip3 install Pillow
RUN pip3 install --upgrade setuptools
RUN pip3 install jupyter
RUN pip3 install pandas
RUN pip3 install datetime
RUN pip3 install tensorflow
RUN pip3 install BeautifulSoup4
RUN pip3 install requests
RUN pip3 install lxml
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install numpy
RUN pip3 install sklearn
RUN pip3 install matplotlib
RUN pip3 install wandb
RUN pip3 install lime
RUN pip3 install seaborn
