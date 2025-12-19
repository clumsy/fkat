#!/bin/bash

python -m ipykernel install --user --name fkat-dev --display-name "FKAT Dev"

cleanup() {
    echo "Shutting down Jupyter notebook..."
    if [ ! -z "$JUPYTER_PID" ]; then
        kill $JUPYTER_PID 2>/dev/null
        wait $JUPYTER_PID 2>/dev/null
    fi
    jupyter kernelspec uninstall -y fkat-dev 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

jupyter notebook examples/hf.ipynb &
JUPYTER_PID=$!

wait $JUPYTER_PID
