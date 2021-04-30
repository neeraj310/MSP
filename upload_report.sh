# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

rm -rf ./report/*
cp -R ../msc-project/* ./report/
git add *
git commit -m "Fix #2"
git push uzh master
