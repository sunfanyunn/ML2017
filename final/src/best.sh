echo $1
mkdir CSVs
for i in 2 4 234 3451 34
do 
    python main.py $i
done
python combine.py $1
