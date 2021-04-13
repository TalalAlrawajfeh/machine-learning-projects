#!/bin/bash

gcc -std=c11 -Wall -O3 -o example matrix.c dnn.c example.c -lm
