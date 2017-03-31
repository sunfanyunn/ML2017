#!/usr/bin/env bash
for i in 1.4 1.45 1.5 1.55 1.6 1.65 1.7 7.75 1.8 1.9 2
do
    echo $i
    echo $(python3 generative.py $i)
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
