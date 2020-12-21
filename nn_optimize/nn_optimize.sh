#!/usr/bin/env bash

root='./data'
gen=200
pop=50
indv=10000
sel='pwts'
elite=1
c_alg='pointwise'
c_prob=1.0
c_point=4
m_prob=0.02
seed=0

# single-gpu
python nn_optimize.py \
--iter 2 --root ${root} --gen ${gen} --pop ${pop} --indv ${indv} --sel ${sel} --elite ${elite} \
--c_alg ${c_alg} --c_prob ${c_prob} --c_point ${c_point} --m_prob ${m_prob} --seed ${seed}

# multi-gpu
# gpu=0,1,2,3
# num_gpu=${gpu//[^0-9]}
# CUDA_VISIBLE_DEVICES=${gpu}
# python -m torch.distributed.launch --nproc_per_node=${#num_gpu} nn_optimize.py \
# --iter 1 --root ${root} --gen ${gen} --pop ${pop} --indv ${indv} --sel ${sel} --elite ${elite} \
# --c_alg ${c_alg} --c_prob ${c_prob} --c_point ${c_point} --m_prob ${m_prob} --seed ${seed}

# SET=$(seq 1 30)
# for i in $SET
# do
# 	echo "Running loop "${i}
# 	python nn_optimize.py --iter ${i} --root ${root} --gen ${gen} --pop ${pop} --indv ${indv} --sel ${sel} --elite ${elite} --c_alg ${c_alg} --c_prob ${c_prob} --c_point ${c_point} --m_prob ${m_prob} --seed ${seed}
# done