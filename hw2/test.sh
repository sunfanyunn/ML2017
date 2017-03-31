#!/usr/bin/env bash
for i in 0.0008 0.0009 0.001 0.0011 0.0012 0.0015 0.0018 0.002
do
    echo $(python3 main.py $i)
done
: '
while ((1))
do 

    num=$(( (RANDOM%106) + 1 ))
    for (( i=0; i<=num; i++ ))
    do
        arr[$i]=$(( (RANDOM%106) + 1 ))
    done

    echo $num, ${arr[@]}

    res=$(python main.py)
    best=0
    if ((res > best)); then
        best=$res
        echo ${arr[@]} >> tmpfile
    fi

done
'
