#!/usr/bin/env bash
for i in 0.01 0.1 1 10 100 1000
do
    # When I execute this script, the python code is modified to accept an
    # argument that represents lambda
    res=$(python3 q4.py $i )
    echo $res >> tmpfile
done
