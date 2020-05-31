#!/bin/bash

docker run \
    --privileged \
    -d -ti \
    --name poetry-demo4 \
    -v /Users/tsubasa/Docker/20200424_poetryenv4:/home/1izard \
    -w /home/1izard \
    devuntu:18.04 \
    /bin/bash
