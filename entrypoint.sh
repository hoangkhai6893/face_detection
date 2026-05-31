#!/bin/bash
set -euo pipefail

USER_NAME="${USER_NAME:-ubuntu}"
WORKDIR="/home/${USER_NAME}/workspace"


if [ -d "$WORKDIR" ]; then
    cd "$WORKDIR"
fi

# If no command is provided, keep the container alive for Dev Containers.
if [ "$#" -eq 0 ]; then
    set -- sleep infinity
fi

exec "$@"