FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    gnupg2 \
    lsb-release \
    locales \
    python3 \
    python3-pip \
    python3-full \
    python3-dev \
    sudo \
    x11-apps \
    zenity \
    openssh-client \
    wget \
    unzip \
    pkg-config \
    zip \
    libxinerama-dev \
    libxcursor-dev \
    xorg-dev \
    libglu1-mesa-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
# --- User Setup
ARG USER_UID
ARG USER_GID
ARG USERNAME

RUN set -eux; \
# Remove user with same UID if exists
EXISTING_USER_BY_UID=$(getent passwd "${USER_UID}" | cut -d: -f1 || true); \
if [ -n "$EXISTING_USER_BY_UID" ]; then \
userdel -r "$EXISTING_USER_BY_UID" || true; \
fi; \
\
# Remove user with same name if exists
if id "${USERNAME}" >/dev/null 2>&1; then \
userdel -r "${USERNAME}" || true; \
fi; \
\
# Remove group with same GID if exists
if getent group "${USER_GID}" >/dev/null; then \
groupdel "$(getent group "${USER_GID}" | cut -d: -f1)"; \
fi; \
\
# Remove group with same name if exists
if getent group "${USERNAME}" >/dev/null; then \
groupdel "${USERNAME}"; \
fi; \
\
# Now create group and user cleanly
groupadd --gid "${USER_GID}" "${USERNAME}"; \
useradd -m -u "${USER_UID}" -g "${USER_GID}" "${USERNAME}"; \
usermod -aG sudo "${USERNAME}"; \
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set environment
ENV USER=${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR /home/${USERNAME}

USER ${USERNAME}

RUN git clone https://github.com/microsoft/vcpkg.git ${HOME}/vcpkg && \
    cd ${HOME}/vcpkg && ./bootstrap-vcpkg.sh -disableMetrics

RUN echo 'export VCPKG_ROOT=$HOME/vcpkg' >> $HOME/.bashrc && \
    echo 'export PATH=$VCPKG_ROOT:$PATH' >> $HOME/.bashrc

RUN wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu128.zip -d ${HOME} && \
    rm libtorch-cxx11-abi-shared-with-deps-2.7.1+cu128.zip

RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.sh && \
    chmod +x cmake-3.29.3-linux-x86_64.sh && \
    sudo ./cmake-3.29.3-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.29.3-linux-x86_64.sh

RUN echo 'export Torch_DIR=$HOME/libtorch/share/cmake/Torch' >> $HOME/.bashrc && \
    echo 'export CMAKE_PREFIX_PATH=$Torch_DIR:$CMAKE_PREFIX_PATH' >> $HOME/.bashrc

