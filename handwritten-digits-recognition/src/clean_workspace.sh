#!/usr/bin/bash

if [[ -d ./data/train ]]
then
    rm -rf ./data/train
fi

if [[ -d ./data/validation ]]
then
    rm -rf ./data/validation
fi

if [[ -d ./data/test ]]
then
    rm -rf ./data/test
fi

find . -maxdepth 1 -name '*.h5' -delete
