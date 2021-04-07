evaluate () {
    local id=$1
    python3 examples/1d_evaluate.py data/1d_lognormal_190000000.csv > uniform_190m_$i.log
}

for i in {1..3};
do 
    evaluate "$i" &
done