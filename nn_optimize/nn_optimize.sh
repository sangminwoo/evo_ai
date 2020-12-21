#!/usr/bin/env bash

root='./data'
gen=400
pop=200
indv=100
sel='pwts'
c_alg='pointwise'
c_prob=1.0
c_point=3
m_prob=0.02
seed=0

python nn_optimize.py --iter 1 --root ${root} --gen ${gen} --pop ${pop} --indv ${indv} --sel ${sel} --c_alg ${c_alg} --c_prob ${c_prob} --c_point ${c_point} --m_prob ${m_prob} --seed ${seed}

# SET=$(seq 1 30)
# for i in $SET
# do
# 	echo "Running loop "${i}
# 	python nn_optimize.py --iter ${i} --root ${root} --gen ${gen} --pop ${pop} --indv ${indv} --sel ${sel} --c_alg ${c_alg} --c_prob ${c_prob} --c_point ${c_point} --m_prob ${m_prob} --seed ${seed}
# done