#!/usr/bin/env bash
while ((1))
do
    for i in 100 500 1000 3000 5000;
    do
        # When I execute this script, the python script is modified to accept an
        # argument that represents how many month of data should the regression
        # use in a single iteration
        res=$(python3 q2.py $i )
        echo $res >> tmpfile
    done
done
