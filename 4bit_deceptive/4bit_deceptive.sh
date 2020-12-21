#!/usr/bin/env bash

gen=400
pop=200
indv=100
sel='pwts'
c_alg='pointwise'
c_prob=1.0
c_point=3
m_prob=0.02

SET=$(seq 1 30)
for i in $SET
do
	echo "Running loop "${i}
	python 4bit_deceptive.py --iter ${i} --gen ${gen} --pop ${pop} --indv ${indv} --sel ${sel} --c_alg ${c_alg} --c_prob ${c_prob} --c_point ${c_point} --m_prob ${m_prob}
done