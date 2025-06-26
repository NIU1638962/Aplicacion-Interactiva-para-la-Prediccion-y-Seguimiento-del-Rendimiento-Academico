#!/bin/bash
configuration_file="./configuration.json"
server_port=$(jq -r '.server_port' "$configuration_file")

if [[ -z "$server_port" || "$server_port" == "null" ]]; then
    echo "Error: Could not read 'server_port' from $configuration_file"
    exit 1
fi

echo "Server port from configuration.json: $server_port"

echo "Execute delete."
bash delete_container.sh

echo "Run server container."
container_id=$(docker run -d --name server -p "$server_port:$server_port" server)
echo "Server container running. ID: $container_id"