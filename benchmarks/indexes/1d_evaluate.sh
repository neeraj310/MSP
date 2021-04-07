# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

evaluate() {
    local id=$1
    local numberOfData=$2
    python3 src/utilities/1d_generator.py uniform $numberOfData
    python3 src/utilities/1d_generator.py normal $numberOfData
    python3 src/utilities/1d_generator.py lognormal $numberOfData

    python3 examples/1d_evaluate.py data/1d_uniform_$numberOfData.csv > uniform_$numberOfData_$i.log
    python3 examples/1d_evaluate.py data/1d_normal_$numberOfData.csv > normal_$numberOfData_$i.log
    python3 examples/1d_evaluate.py data/1d_lognormal_$numberOfData.csv > lognormal_$numberOfData_$i.log
}
for i in {1..3};
do
    evaluate "$i" "10000" &
done