FROM python:3.13-slim


ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

SHELL ["/bin/bash","-o" ,"pipefail","-c"]

RUN DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 

RUN apt-get update -y
RUN apt-get install -y wget curl git tmux imagemagick htop libsndfile1 nfs-common unzip cmake libgl1 python3-opencv sudo 
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs
RUN npm install -g opencode-ai @anthropic-ai/claude-code
RUN  apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --only-binary=:all: --no-cache-dir -r requirements.txt || \
    (sed -i '/^dlib/d; /^face-recognition/d' requirements.txt && pip install --no-cache-dir -r requirements.txt)

# RUN apt-get update && apt-get install -y sudo  
# Since uid and gid will change at entrypoint, anything can be used

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER_NAME=dkhai
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -d /home/${USER_NAME} -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}  | chpasswd
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}



# COPY --chown=${USER_NAME}:${USER_NAME} ./99-intrepidcs.rules /etc/udev/rules.d/99-intrepidcs.rules

COPY --chown=${USER_NAME}:${USER_NAME} --chmod=777 ./entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh
WORKDIR /home/${USER_NAME}/workspace

ENTRYPOINT ["/entrypoint.sh"]

