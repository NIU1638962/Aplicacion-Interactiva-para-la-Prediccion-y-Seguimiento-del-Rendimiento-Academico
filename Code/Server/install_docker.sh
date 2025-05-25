#!/bin/bash

if command -v docker > /dev/null 2>&1; then
    echo "Docker is already installed. Skipping installation."
else
    echo "Docker is not installed. Proceeding with installation."

    echo "Updating package list."
    sudo apt-get update -y

    echo "Installing prerequisites."
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    echo "Adding Docker's GPG key."
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo "Adding Docker's stable repository."
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    echo "Updating package list with Docker's repository."
    sudo apt-get update -y

    echo "Installing Docker Engine."
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "Verifying Docker installation."
    sudo docker --version

    echo "Starting and enabling Docker service."
    sudo systemctl enable docker
    sudo systemctl start docker

    sudo service docker start
    sudo service docker status

    echo "Docker installation completed."
fi
