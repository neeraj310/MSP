# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

for i in {1..5}
do
    # first generate the data
    python3 src/utilities/1d_generator.py uniform 10000
    python3 src/utilities/1d_generator.py normal 10000
    python3 src/utilities/1d_generator.py lognormal 10000

    python3 examples/1d_evaluate.py data/1d_uniform_10000.csv > uniform_$i.log
    python3 examples/1d_evaluate.py data/1d_normal_10000.csv > normal_$i.log
    python3 examples/1d_evaluate.py data/1d_lognormal_10000.csv > lognormal_$i.log
done