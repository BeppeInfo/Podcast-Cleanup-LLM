#!/usr/bin/env bash
#
# Which of the two things this image is, decided by the first argument.
#
#   web            serve the interface        (the default)
#   cli [args...]  run clean-podcast.sh       (one episode, then exit)
#   test           run both suites
#   anything else  exec it, so `docker run … bash` works
#
# The pipeline is the same code either way; this only chooses the front end.

set -euo pipefail

PODCAST_ROOT="${PODCAST_ROOT:-/data}"
mkdir -p "$PODCAST_ROOT"/{incoming,output,work,failed}

warn_if_unset() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        printf 'warning: %s is not set. Both models are servers reached over\n' "$name" >&2
        printf '         HTTP and none is in this image — pass -e %s=http://host:port\n' "$name" >&2
    fi
}

case "${1:-web}" in
    web)
        warn_if_unset WHISPER_ENDPOINT
        [[ "${LLM_ENABLE:-1}" == 1 ]] && warn_if_unset LLAMA_ENDPOINT
        # One process, on purpose. The single-job lock lives in this process's
        # memory, so a second worker would be a second lock and two episodes
        # could run at once over one work directory. Threads are fine — they
        # share it — which is why this is waitress and not gunicorn with
        # --workers 2.
        exec python -m waitress \
            --host="${PODCAST_WEB_HOST:-0.0.0.0}" \
            --port="${PODCAST_WEB_PORT:-8000}" \
            --threads="${PODCAST_WEB_THREADS:-4}" \
            --call web.app:create_app
        ;;
    cli)
        shift
        exec /app/clean-podcast.sh "$@"
        ;;
    test)
        cd /app
        python -m pytest tests/test_pipeline.py -q
        exec ./tests/selftest.sh
        ;;
    *)
        exec "$@"
        ;;
esac
