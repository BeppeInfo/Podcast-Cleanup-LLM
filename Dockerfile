# Podcast cleanup, containerised.
#
# Three things this image is for, in order:
#
# 1. Pinning the interpreter. The pipeline itself is standard-library Python and
#    runs on anything current, but the machine this was written on is on 3.14,
#    which several audio libraries do not support yet — WhisperX caps at <3.14.
#    Pinning here means the host's Python stops being a constraint.
#
# 2. Carrying WhisperX. Transcription is no longer a server, so it has to be
#    installed somewhere, and the checkout is deliberately dependency-free.
#
# 3. Making the layout somebody else's problem. One volume at /data, everything
#    under it, and settings by environment.
#
# **No config file is shipped.** `config_load` searches for one and, finding
# none, lets the environment through — which is the precedence a container
# wants. Mounting a podcast-cleanup.conf into the image would silently outrank
# every -e you pass, so do not, unless that is what you mean.

ARG PYTHON_VERSION=3.13

# --- base: the interpreter, ffmpeg, and the pipeline itself -------------------
FROM python:${PYTHON_VERSION}-slim AS base

# ffmpeg does all the audio; ffprobe comes with it. The detector is still a
# server reached over HTTP and is not in this image; transcription now is.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PODCAST_ROOT=/data

# Weights land here rather than in root's home, so one mounted volume keeps both
# the whisper model and the wav2vec2 aligner across rebuilds.
ENV HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

# --- whisperx: CPU only, and pinned in three places for three reasons --------
#
# torch comes from the CPU index because the default resolves the CUDA build and
# drags in several gigabytes of nvidia-* wheels. Neither available machine has an
# NVIDIA card — both are Radeon — so those would be dead weight that never runs.
#
# torch is pinned to 2.8.0 because whisperx 3.8.6 requires exactly that. Left
# unpinned, pip takes the newest CPU torch and then silently resolves *whisperx*
# backwards to 3.7.2 to fit it, with no error and no warning.
#
# torchvision has to come from the CPU index too. Nothing here wants vision —
# pyannote imports it — and the PyPI build is compiled against the other ABI, so
# it fails at import with "operator torchvision::nms does not exist".
#
# The constraint file is what stops whisperx's own resolve from undoing all of
# the above while installing its dependencies.
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG WHISPERX_VERSION=3.8.6
RUN pip install --no-cache-dir \
      torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip freeze | grep -E '^(torch|torchaudio|torchvision)==' > /tmp/cpu-torch.txt \
 && pip install --no-cache-dir whisperx==${WHISPERX_VERSION} \
      --constraint /tmp/cpu-torch.txt \
 && rm /tmp/cpu-torch.txt

WORKDIR /app
COPY clean-podcast.sh ./
COPY lib/ ./lib/
COPY python/ ./python/
COPY docker-entrypoint.sh /usr/local/bin/

RUN chmod +x clean-podcast.sh /usr/local/bin/docker-entrypoint.sh \
 && mkdir -p /data/incoming /data/output /data/work /data/failed /models

# --- runtime: what you deploy -------------------------------------------------
FROM base AS runtime

# Flask for the interface, waitress to serve it. Deliberately nothing else:
# numpy and scipy belong to tools/, which is for measuring a run against a hand
# edit and has no business in a production image.
COPY web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY web/ ./web/

VOLUME ["/data", "/models"]
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
