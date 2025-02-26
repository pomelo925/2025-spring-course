#!/bin/bash

export DISPLAY=:0

cd docker
docker compose -p robotic-nav-exploration up -d