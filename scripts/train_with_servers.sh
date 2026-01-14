#!/bin/bash
set -e

cd examples

# Cleanup function
cleanup() {
    echo "Stopping servers..."
    pkill -f "tensorboard --logdir" 2>/dev/null || true
    pkill -f "mlflow ui" 2>/dev/null || true
    wandb server stop 2>/dev/null || true
    exit
}

trap cleanup INT TERM EXIT

# Run training
PYTHONPATH=. python ../fkat/train.py -cd conf -cn hf "$@"

# Start servers
mlflow ui --backend-store-uri file://$(pwd)/mlflow --host 0.0.0.0 --port 5000 &
tensorboard --logdir=tensorboard --host=0.0.0.0 --port=6006 &

sleep 3
open http://localhost:5000
open http://localhost:6006

# Start wandb if Docker is available
if docker info >/dev/null 2>&1; then
    echo "Starting wandb server..."
    wandb server start --host=0.0.0.0 --port=8080 &

    # Wait for wandb to be ready
    while true; do
        sleep 2
        if curl -s http://localhost:8080 >/dev/null 2>&1; then
            echo "Wandb server started"
            open http://localhost:8080
            echo
            echo "Get API key from http://localhost:8080/authorize"
            read -p "Enter wandb API key (or press Enter to skip): " WANDB_KEY

            if [ -n "$WANDB_KEY" ]; then
                echo "Syncing offline runs..."
                export WANDB_BASE_URL=http://localhost:8080
                export WANDB_API_KEY="$WANDB_KEY"

                # Try standard wandb sync first (handles .wandb files)
                for run_dir in wandb/wandb/offline-run-*; do
                    if [ -d "$run_dir" ]; then
                        echo "Syncing $run_dir..."
                        wandb sync "$run_dir" 2>&1 | grep -v "Exception in thread" | grep -v "Traceback" | grep -v "FileNotFoundError" || true
                    fi
                done

                # Then sync any with summary files using our custom script
                python sync_wandb_offline.py http://localhost:8080 "$WANDB_KEY" wandb/wandb/offline-run-* 2>&1 | grep -v Traceback | grep -v File
                echo "Synced!"
            fi
            break
        fi
    done
else
    echo "WARNING: Docker not running - wandb server requires Docker"
fi

echo
echo "All servers running. Press Ctrl+C to stop all servers."
wait
