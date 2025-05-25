#!/bin/bash
echo "Execute stop."
bash stop_container.sh

echo "Removing server container."
echo "Checking if server container exists and stopped."
output=$(docker container ls -f status=exited -f name=server -a | tail -n +2 | head -1 | wc -m)

if [ "$output" -gt 0 ];
then
    echo "Does exist server container that is stopped. Removing server container."
    docker remove server  > /dev/null 2>&1
    echo "Server container removed."
else
    echo "Does not exist server container that is stopped."
fi