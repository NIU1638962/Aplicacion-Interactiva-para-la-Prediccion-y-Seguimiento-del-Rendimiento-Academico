#!/bin/bash

echo "Preparing environment."
if [ -d "Aplicacion-Interactiva-para-la-Prediccion-y-Seguimiento-del-Rendimiento-Academico" ];
then
    echo "Directory already exists. Removing."
    rm -d -f -r Aplicacion-Interactiva-para-la-Prediccion-y-Seguimiento-del-Rendimiento-Academico
    echo "Directory removed."
fi

if command -v git > /dev/null 2>&1; then
    echo "Git is already installed."
else
    echo "Git is not installed. Installing git."
    sudo apt-get update -y
    sudo apt-get install -y git
    echo "Git has been installed."
fi

echo "Cloning repository with the server infrastructure."
git clone https://github.com/NIU1638962/Aplicacion-Interactiva-para-la-Prediccion-y-Seguimiento-del-Rendimiento-Academico.git
echo "Repository cloned."

echo "Setting working directory."
cd Aplicacion-Interactiva-para-la-Prediccion-y-Seguimiento-del-Rendimiento-Academico/Code/Server
echo "Working directory set."

echo "Execute install docker."
bash install_docker.sh
echo "Environment prepared."

echo "Execute deploy."
bash deploy_container.sh