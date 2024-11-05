#!/bin/bash

CUR_DIR=`pwd`

for i in {1..7}
do
    if [ -d "toy_chapter_0${i}/build" ]; then
        rm -rf "toy_chapter_0${i}/build"
    fi
done

