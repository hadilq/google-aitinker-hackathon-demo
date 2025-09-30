FROM python:3.12

# Make the version configurable
# https://github.com/hashicorp/terraform/releases
ARG TERRAFORM_VERSION=1.13.3

# Install packages available through apt
RUN apt-get update && apt-get install -y \
  bash-completion \
  curl \
  ca-certificates \
  git \
  unzip \
  libgl1 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Terraform as root user
RUN curl -fsSL https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip -o /tmp/terraform.zip \
  && unzip /tmp/terraform.zip -d /tmp \
  && mv /tmp/terraform /usr/local/bin/ \
  && rm -rf /tmp/* \
  && terraform version

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

RUN adduser --home /home/dev/src dev

ENV PATH="/root/.local/bin:/home/dev/src/.local/bin:${PATH}"
WORKDIR /home/dev/src

RUN uv venv

COPY sam_agent/requirements.txt /requirements.txt
RUN uv pip sync /requirements.txt

# Switch to non-root user
USER dev

# Install auto-completions for non-root user
RUN terraform -install-autocomplete

CMD ["python"]
