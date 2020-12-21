#!/usr/bin/env bash

iter=1
gen=10000
pop=300
sel='pwts'
c_prob=0.9
c_point=3
m_prob=0.01

SET=$(seq 1 30)
for i in $SET
do
	echo "Running loop "${i}
	python salesman.py --iter ${i} --gen ${gen} --pop ${pop} --sel ${sel} --c_prob ${c_prob} --c_point ${c_point} --m_prob ${m_prob}
done