# Podcast cleanup, containerised.
#
# Two things this image is for, in order:
#
# 1. Pinning the interpreter. The pipeline itself is standard-library Python and
#    runs on anything current, but the machine this was written on is on 3.14,
#    which several audio libraries do not support yet — WhisperX caps at <3.14.
#    Pinning here means the host's Python stops being a constraint.
#
# 2. Making the layout somebody else's problem. One volume at /data, everything
#    under it, and settings by environment.
#
# **No config file is shipped.** `config_load` searches for one and, finding
# none, lets the environment through — which is the precedence a container
# wants. Mounting a podcast-cleanup.conf into the image would silently outrank
# every -e you pass, so do not, unless that is what you mean.

ARG PYTHON_VERSION=3.12

# --- base: the interpreter, ffmpeg, and the pipeline itself -------------------
FROM python:${PYTHON_VERSION}-slim AS base

# ffmpeg does all the audio; ffprobe comes with it. Nothing else is needed —
# both models are servers reached over HTTP and are not in this image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PODCAST_ROOT=/data

WORKDIR /app
COPY clean-podcast.sh ./
COPY lib/ ./lib/
COPY python/ ./python/
COPY docker-entrypoint.sh /usr/local/bin/

RUN chmod +x clean-podcast.sh /usr/local/bin/docker-entrypoint.sh \
 && mkdir -p /data/incoming /data/output /data/work /data/failed

# --- runtime: what you deploy -------------------------------------------------
FROM base AS runtime

# Flask for the interface, waitress to serve it. Deliberately nothing else:
# numpy and scipy belong to tools/, which is for measuring a run against a hand
# edit and has no business in a production image.
COPY web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY web/ ./web/

VOLUME ["/data"]
EXPOSE 8000

# Nothing here has authentication and ffmpeg is handed whatever is uploaded, so
# bind to the loopback of whatever network you put it on, not the internet.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["web"]

# --- dev: the suites and the measuring tools ----------------------------------
FROM runtime AS dev

# numpy and scipy are only for tools/recover_cuts.py, which aligns a hand edit
# against its original. bash and curl are for tests/selftest.sh.
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir numpy scipy pytest

COPY tests/ ./tests/
COPY tools/ ./tools/
COPY podcast-cleanup.conf.example DESIGN.md README.md ./

CMD ["test"]
