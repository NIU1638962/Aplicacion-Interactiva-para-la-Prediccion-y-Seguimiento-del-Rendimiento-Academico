#!/bin/bash
echo "Deploying server container."

echo "Execute build."
bash build_container.sh

echo "Execute run."
bash run_container.sh

echo "Server container deployed."
