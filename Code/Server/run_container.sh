#!/bin/bash
echo "Execute delete."
bash delete_container.sh

echo "Run server container."
container_id=$(docker run -d --name server -p 8080:8080 server)
echo "Server container running. ID: $container_id"S