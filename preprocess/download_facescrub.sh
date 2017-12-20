#!/bin/bash

# Compile and run download facescrub program
g++ -o download_facescrub -Wall -std=c++14 -g download_facescrub.cpp \
 -D DLIB_JPEG_SUPPORT -DDLIB_PNG_SUPPORT -DDLIB_GIF_SUPPORT \
 -I /home/sourabhd/installs/dlib -I /home/sourabhd/.local/include \
 -L /home/sourabhd/installs/dlib/build/dlib \
 -L /home/sourabhd/.local/lib -L /usr/lib/x86_64-linux-gnu \
 -fopenmp -lboost_system -lboost_filesystem -lcurl -lpthread -ljpeg -lpng -lgif -ldlib \
 && OMP_PROC_BIND=true OMP_NUM_THREADS=64 ./download_facescrub
