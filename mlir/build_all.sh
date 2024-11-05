#!/bin/bash

CUR_DIR=`pwd`

for i in {1..7}
do
    mkdir "toy_chapter_0${i}/build"
    cd "toy_chapter_0${i}/build"
    cmake -G Ninja ..

    ninja
    cd $CUR_DIR
done

