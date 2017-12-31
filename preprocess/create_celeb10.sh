#!/bin/bash

basedir="/data/datasets/facescrub_images"
num=5
dset="celeb10"

for s in "actors" "actresses"
do
    cd "$basedir/$s/faces"
    subset="$(for x in `ls -1` ; do echo "$(ls -1 $x | wc -l ) $x" ; done | sort -n | awk '{print $2}' | tail -n 5 | xargs)"
    destdir="$basedir/$dset"
    mkdir -p $destdir
    mkdir -p $destdir/train
    mkdir -p $destdir/val
    echo $subset
    for d in $subset
    do
      lines="$(ls -1 $d | shuf)"
      for y in $(echo $lines| sed "s/ /\n/g" | head -n 100)
      do
            mkdir -p $destdir/train/$d
            #  echo "$basedir/$s/faces/$d/$y $destdir/train/$d/$y"
            cp -av $basedir/$s/faces/$d/$y $destdir/train/$d/$y
      done
      for y in $(echo $lines | sed "s/ /\n/g" | sed "1,100d")
      do
            mkdir -p $destdir/val/$d
            # echo "$basedir/$s/faces/$d/$y $destdir/val/$d/$y"
            cp -av $basedir/$s/faces/$d/$y $destdir/val/$d/$y
      done
    done
    cd -
done
