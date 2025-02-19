FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fish \
    git \
    curl \
    vim \
    bash \
    ffmpeg \
    tmux \
    unzip \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# cargo \
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
#     source $HOME/.cargo/env && \
#     rustup default stable

# RUN rustup install stable && rustup default stable

# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
#     source $HOME/.cargo/env && \
#     rustup default stable


# Declare build arguments before using them
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create a group and user with the specified UID and GID
RUN addgroup --gid $GROUP_ID appgroup && \
    adduser --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --disabled-password --gecos "" appuser

# Install sudo
RUN apt-get update && apt-get install -y sudo

# Grant sudo privileges to appuser without a password
RUN echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Switch to the new user
USER appuser

SHELL ["/usr/bin/fish", "-c"]

WORKDIR /app
# Create the virtual environment in a directory not affected by the mount
# Copy requirements.txt and install dependencies
# COPY --chown=appuser:appgroup requirements.txt ./
# RUN uv venv && uv pip install --upgrade pip && \
#     uv pip install -r requirements.txt


# Copy the entire project into the container
COPY --chown=appuser:appgroup pyproject.toml ./
COPY --chown=appuser:appgroup uv.lock ./

RUN uv sync

EXPOSE 8000

# CMD is now managed by docker-compose to run the Fish shell
