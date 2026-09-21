FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships a Yarn apt repo with an outdated signing key, which makes
# `apt-get update` fail (breaking the docker-in-docker feature). We don't use Yarn.
RUN grep -rl "dl.yarnpkg.com" /etc/apt/sources.list.d/ 2>/dev/null | xargs -r rm -f
