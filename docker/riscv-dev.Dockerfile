ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt-get update && \
    apt-get install software-properties-common wget && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    add-apt-repository --yes --no-update "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/llvm.asc && \
    apt-get update && \
    # install compilers to build OpenVINO for RISC-V 64
    apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu && \
    apt-get install \
        curl \
        git \
        cmake \
        ccache \
        ninja-build \
        fdupes \
        patchelf \
        ca-certificates \
        gpg-agent \
        tzdata \
        pkg-config \
        # parallel gzip
        pigz \
        # Python \
        python3 \
        python3-dev \
        python3-venv \
        # Compilers
        gcc \
        g++ \
        # riscv-gnu-toolchain build dependencies
        autoconf \
        automake \
        autotools-dev \
        libmpc-dev \
        libmpfr-dev \
        libgmp-dev \
        gawk \
        build-essential \
        bison \
        flex \
        texinfo \
        gperf \
        libtool \
        patchutils \
        bc \
        zlib1g-dev \
        libglib2.0-dev \
        libslirp-dev \
        libncurses-dev \
        libexpat-dev \
        python3-tomli \
        # For clang-tidy validation
        clang-format-18 \
        clang-tidy-18 \
        libomp-18-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install RISC-V native debian packages
RUN echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy main restricted > riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy universe >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates universe >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy multiverse >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates multiverse >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse >> riscv64-sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security main restricted >> riscv64-sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security universe >> riscv64-sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security multiverse >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy main >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy universe >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-security main >> riscv64-sources.list && \
    mv riscv64-sources.list /etc/apt/sources.list.d/

RUN dpkg --add-architecture riscv64 && \
    apt-get update -o Dir::Etc::sourcelist=/etc/apt/sources.list.d/riscv64-sources.list && \
    apt-get install -y --no-install-recommends libpython3-dev:riscv64

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# build riscv-collab toolchain
ARG RISCV_GNU_TOOLCHAIN_REF="2026.03.13"
ARG RISCV_GNU_TOOLCHAIN_REPO="https://github.com/riscv-collab/riscv-gnu-toolchain.git"
ARG RISCV_TOOLCHAIN_PATH="/opt/riscv"
ARG RISCV_TOOLCHAIN_TMP_PATH="/tmp/riscv-gnu-toolchain"
ARG RISCV_TOOLCHAIN_SRC="/tmp/riscv-gnu-toolchain/src"

RUN mkdir -p ${RISCV_TOOLCHAIN_TMP_PATH} && cd ${RISCV_TOOLCHAIN_TMP_PATH} && \
    git clone --branch ${RISCV_GNU_TOOLCHAIN_REF} --depth 1 ${RISCV_GNU_TOOLCHAIN_REPO} ${RISCV_TOOLCHAIN_SRC} && \
    cd ${RISCV_TOOLCHAIN_SRC} && \
    ./configure --prefix=${RISCV_TOOLCHAIN_PATH} && \
    make -j"$(nproc)" linux build-qemu && \
    rm -rf ${RISCV_TOOLCHAIN_TMP_PATH}
