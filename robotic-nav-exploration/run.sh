#!/bin/bash

export DISPLAY=:0

xhost +local:docker

cd docker
docker compose -p robotic-nav-exploration up -d