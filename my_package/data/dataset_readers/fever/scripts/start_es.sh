#!/bin/bash

docker run -p 9200:9200 \
    -e "discovery.type=single-node" \
    --name elastic-wiki \
    -d \
    docker.elastic.co/elasticsearch/elasticsearch:7.14.1