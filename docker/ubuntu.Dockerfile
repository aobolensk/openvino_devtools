# Unified OpenVINO *build* image for
#   * x86‑64             (native)
#   * aarch64 / arm64    (cross‑compile)
#   * riscv64            (cross‑compile)
#
# ‑ Ubuntu 24.04 base
# ‑ Clang/LLVM 18 tool‑chain (clang‑tidy‑18, clang‑format‑18 …)
#
# This merges the essential build‑time dependencies from the
# original OpenVINO builder images:
#   • .github/dockerfiles/ov_build/ubuntu_20_04_x64/Dockerfile
#   • .github/dockerfiles/ov_build/ubuntu_22_04_arm64/Dockerfile
#   • .github/dockerfiles/ov_build/ubuntu_22_04_riscv/Dockerfile
#   • .github/dockerfiles/ov_build/ubuntu_22_04_riscv_xuantie/Dockerfile
#
# The resulting container can **generate compile commands
# and run clang‑tidy checks for all three targets in one place**.

FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        cmake \
        git \
        wget curl ca-certificates gnupg2 lsb-release \
        python3 python3-pip python3-setuptools python3-wheel \
        pkg-config ccache \
        scons \
        # LLVM/Clang 18
        clang-18 clang-tidy-18 clang-format-18 clang-tools-18 \
        lld-18 llvm-18-dev llvm-18-runtime \
        # Native GCC (x86‑64) comes with build‑essential
        # Cross tool‑chains
        gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
        libc6-dev-arm64-cross \
        gcc-riscv64-linux-gnu g++-riscv64-linux-gnu \
        libc6-dev-riscv64-cross \
        crossbuild-essential-arm64 crossbuild-essential-riscv64 \
        # ---- extra utilities & build deps (from riscv_xuantie image) ----
        fdupes \
        patchelf \
        pigz \
        # Python headers / tooling
        python3-dev \
        python3-venv \
        # Generic autotools / build helpers
        autoconf \
        automake \
        autotools-dev \
        libmpc-dev \
        libmpfr-dev \
        libgmp-dev \
        gawk \
        bison \
        flex \
        texinfo \
        gperf \
        libtool \
        patchutils \
        bc \
        zlib1g-dev \
        libexpat-dev \
        # OpenMP runtime for Clang 18
        libomp-18-dev \
        # User‑mode emulation (optional — enables configure/tests)
        qemu-user qemu-user-static binfmt-support \
        # Misc utilities
        openssh-client sudo tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --no-cache-dir \
    --break-system-packages \
    --ignore-installed \
    --upgrade pip setuptools wheel

ARG TARGETARCH
ARG SCCACHE_VERSION="v0.7.5"

RUN ARCH="${TARGETARCH:-$(uname -m)}" && \
    case "${ARCH}" in \
        amd64|x86_64) echo "x86_64-unknown-linux-musl" > /tmp/sccache_arch ;; \
        arm64|aarch64) echo "aarch64-unknown-linux-musl" > /tmp/sccache_arch ;; \
        *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac

ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"
RUN set -eux; \
    SCCACHE_ARCH=$(cat /tmp/sccache_arch) && \
    mkdir -p ${SCCACHE_HOME} && \
    cd ${SCCACHE_HOME} && \
    curl -sSL -o sccache.tar.gz \
        "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${SCCACHE_ARCH}.tar.gz" && \
    tar -xzf sccache.tar.gz --strip-components=1 && \
    rm sccache.tar.gz /tmp/sccache_arch
ENV PATH="${SCCACHE_HOME}:${PATH}"

ARG XUANTIE_VERSION="V2.8.1"
ARG XUANTIE_REPO="https://github.com/XUANTIE-RV/xuantie-gnu-toolchain"
ARG XUANTIE_PREFIX="/opt/riscv"
RUN set -eux; \
    mkdir -p /tmp/xuantie && \
    git clone --branch "${XUANTIE_VERSION}" --depth 1 "${XUANTIE_REPO}" /tmp/xuantie/src && \
    cd /tmp/xuantie/src && \
    ./configure --prefix="${XUANTIE_PREFIX}" --disable-gdb && \
    make linux -j2 && \
    make install && \
    rm -rf /tmp/xuantie
ENV PATH="${XUANTIE_PREFIX}/bin:${PATH}"

WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-l"]
CMD ["-i"]
