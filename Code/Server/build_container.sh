#!/bin/bash
echo "Building server container."

docker build --tag server .

echo "Server container build."