evaluate() {
    local runid=$1
    python3 src/utilities/1d_generator.py lognormal 10000
    python3 examples/staged_grid_search.py data/1d_lognormal_10000.csv > lognormal_staged_$runid.log
}
for i in {1..2}; do evaluate "$i" & done