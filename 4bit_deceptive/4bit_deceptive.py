import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import random
import matplotlib.pyplot as plt

class Fourbit:
	# Order-4 deceptive problem
	def __init__(self, crossover_algorithm, crossover_prob, crossover_points, mutation_prob,
				 population_size, individual_len, num_generations, selection_algorithm):
		self.crossover_algorithm = crossover_algorithm
		self.crossover_prob = crossover_prob
		self.crossover_points = crossover_points
		self.mutation_prob = mutation_prob
		self.population_size = population_size
		self.individual_len = individual_len
		self.num_generations = num_generations
		self.selection_algorithm = selection_algorithm

	def init_population(self):
		return np.stack([np.random.choice(2, self.individual_len, replace=True) for _ in range(self.population_size*2)]) # 300*2x100

	def fitness(self, string):
		if len(string) != 100:
			raise Exception("Error: The length of every chromosome must be 100")
		fitness = 0
		for i in range(0, 99, 4):
			if sum(string[i:i+4]) == 4:
				fitness += 4
			else:
				fitness += 3 - sum(string[i:i+4])
		return fitness

	def selection_rws(self, fitness, population):
		# Roulette wheel selection
		probability = np.array(fitness) / np.sum(fitness)
		selected = np.random.choice(population.shape[0], self.population_size, p=probability)
		return population[selected]

	def selection_pwts(self, fitness, population):
		# Pair-wise tournament selection
		selected = []
		for i in range(0, population.shape[0], 2):
			if fitness[i] > fitness[i+1]:
				selected.append(i)
			else:
				selected.append(i+1)
		return population[selected]

	def selection_rank(self, fitness, population):
		# Rank-based selection
		selected = np.argsort(fitness)[:self.population_size]
		return population[selected]

	def uniform_crossover(self, parents):
		# Uniform crossover
		children = np.zeros((parents.shape[0]*2, parents.shape[1]), dtype=int)
		
		for i in range(0, children.shape[0], 2):
			idx = np.random.choice(parents.shape[0], 2, replace=False)

			if random.random() < self.crossover_prob:
				mask1 = np.random.choice(2, parents.shape[1], replace=True)
				mask2 = 1-mask1

				children[i, mask1 > 0] = parents[idx[0], mask1 > 0]
				children[i+1, mask2 > 0] = parents[idx[1], mask2 > 0]
			else:
				children[i, :] = parents[idx[0], :]
				children[i+1, :] = parents[idx[1], :]

		return children

	def pointwise_crossover(self, parents):
		# Pointwise crossover
		children = np.zeros((parents.shape[0]*2, parents.shape[1]), dtype=int)
		
		for i in range(0, children.shape[0], 2):
			idx = np.random.choice(parents.shape[0], 2, replace=False)

			if random.random() < self.crossover_prob:
				crossover_points = sorted(np.random.choice(parents.shape[1], self.crossover_points, replace=False))
				crossover_points = [0] + crossover_points + [parents.shape[1]]
				for j in range(0, len(crossover_points)-1, 2):
					children[i, crossover_points[j]:crossover_points[j+1]] = parents[idx[0], crossover_points[j]:crossover_points[j+1]]
					children[i, crossover_points[j+1]:crossover_points[j+2]] = parents[idx[1], crossover_points[j+1]:crossover_points[j+2]]

				for j in range(0, len(crossover_points)-1, 2):
					children[i+1, crossover_points[j]:crossover_points[j+1]] = parents[idx[1], crossover_points[j]:crossover_points[j+1]]
					children[i+1, crossover_points[j+1]:crossover_points[j+2]] = parents[idx[0], crossover_points[j+1]:crossover_points[j+2]]

				# children[i, :crossover_points[0]] = parents[idx[0], :crossover_points[0]]
				# children[i, crossover_points[0]:crossover_points[1]] = parents[idx[1], crossover_points[0]:crossover_points[1]]
				# children[i, crossover_points[1]:crossover_points[2]] = parents[idx[0], crossover_points[1]:crossover_points[2]]
				# children[i, crossover_points[2]:] = parents[idx[1], crossover_points[2]:]

				# children[i+1, :crossover_points[0]] = parents[idx[1], :crossover_points[0]]
				# children[i+1, crossover_points[0]:crossover_points[1]] = parents[idx[0], crossover_points[0]:crossover_points[1]]
				# children[i+1, crossover_points[1]:crossover_points[2]] = parents[idx[1], crossover_points[1]:crossover_points[2]]
				# children[i+1, crossover_points[2]:] = parents[idx[0], crossover_points[2]:]

				# handle duplicate
				for j in [i, i+1]:
					flag = {}
					for idx in range(children.shape[1]):
						flag[idx] = []
					
					for idx, val in enumerate(children[j]):
						flag[val].append(idx)

					missing = [key for key in flag if len(flag[key])==0]
					duplicate = [key for key in flag if len(flag[key])==2]

					for (m, d) in zip(missing, duplicate):
						children[j, flag[d][-1]] = m
			else:
				children[i, :] = parents[idx[0], :]
				children[i+1, :] = parents[idx[1], :]

		return children

	def bbwise_crossover(self, parents):
		def linkage_identification(population, threshold):
			def mutual_info(genes1, genes2):
				eps = 1e-10 # to avoid devision by zero
				population = len(genes1)
				g1_0 = np.sum(genes1==0)/population
				g1_1 = np.sum(genes1==1)/population
				g2_0 = np.sum(genes2==0)/population
				g2_1 = np.sum(genes2==1)/population
				g1_0_g2_0 = np.sum(np.logical_and(genes1==0, genes2==0))/population + eps
				g1_0_g2_1 = np.sum(np.logical_and(genes1==0, genes2==1))/population + eps
				g1_1_g2_0 = np.sum(np.logical_and(genes1==1, genes2==0))/population + eps
				g1_1_g2_1 = np.sum(np.logical_and(genes1==1, genes2==1))/population + eps

				info = g1_0_g2_0*np.log(g1_0_g2_0 / (g1_0*g2_0+eps)) + \
					   g1_0_g2_1*np.log(g1_0_g2_1 / (g1_0*g2_1+eps)) + \
					   g1_1_g2_0*np.log(g1_1_g2_0 / (g1_1*g2_0+eps)) + \
					   g1_1_g2_1*np.log(g1_1_g2_1 / (g1_1*g2_1+eps))
				return info

			linkage = []
			infos = {}
			for i in range(parents.shape[1]):
				for j in range(i+1, parents.shape[1]):
					infos[(i,j)] = mutual_info(parents[:, i], parents[:, j])
					info_idx, max_info = np.argmax(list(infos.values())), np.max(list(infos.values()))
					if max_info > threshold:
						linkage.append(info_idx)

			return linkage

		# links = linkage_identification(parents, threshold=0.1)
		# bb-wise crossover
		children = np.zeros((parents.shape[0]*2, parents.shape[1]), dtype=int)
		
		for i in range(0, children.shape[0], 2):
			idx = np.random.choice(parents.shape[0], 2, replace=False)

			if random.random() < self.crossover_prob:
				crossover_points = sorted(np.random.choice(parents.shape[1]//4, self.crossover_points, replace=False)*4)
				
				children[i, :crossover_points[0]] = parents[idx[0], :crossover_points[0]]
				children[i, crossover_points[0]:crossover_points[1]] = parents[idx[1], crossover_points[0]:crossover_points[1]]
				children[i, crossover_points[1]:crossover_points[2]] = parents[idx[0], crossover_points[1]:crossover_points[2]]
				children[i, crossover_points[2]:] = parents[idx[1], crossover_points[2]:]

				children[i+1, :crossover_points[0]] = parents[idx[1], :crossover_points[0]]
				children[i+1, crossover_points[0]:crossover_points[1]] = parents[idx[0], crossover_points[0]:crossover_points[1]]
				children[i+1, crossover_points[1]:crossover_points[2]] = parents[idx[1], crossover_points[1]:crossover_points[2]]
				children[i+1, crossover_points[2]:] = parents[idx[0], crossover_points[2]:]

				# handle duplicate
				for j in [i, i+1]:
					flag = {}
					for idx in range(children.shape[1]):
						flag[idx] = []
					
					for idx, val in enumerate(children[j]):
						flag[val].append(idx)

					missing = [key for key in flag if len(flag[key])==0]
					duplicate = [key for key in flag if len(flag[key])==2]

					for (m, d) in zip(missing, duplicate):
						children[j, flag[d][-1]] = m
			else:
				children[i, :] = parents[idx[0], :]
				children[i+1, :] = parents[idx[1], :]

		return children

	def mutation(self, children):
		# pair-wise mutation
		for i in range(children.shape[0]):
			if random.random() < self.mutation_prob:
				mutation_points = np.random.choice(children.shape[1], 2, replace=False)
				temp = children[i, mutation_points[0]]
				children[i, mutation_points[0]] = children[i, mutation_points[1]]
				children[i, mutation_points[1]] = temp
		return children

	def run(self):
		fitness_history = []
		population = self.init_population()

		for i in range(self.num_generations):
			fitness = np.array([self.fitness(indv) for indv in population])
			fitness_history.append(fitness)
			
			# selection
			if self.selection_algorithm == 'rws':
				parents = self.selection_rws(fitness, population)
			elif self.selection_algorithm == 'pwts':
				parents = self.selection_pwts(fitness, population)
			elif self.selection_algorithm == 'rank':
				parents = self.selection_rank(fitness, population)

			# crossover
			if self.crossover_algorithm == 'uniform':
				children = self.uniform_crossover(parents)
			elif self.crossover_algorithm == 'pointwise':
				children = self.pointwise_crossover(parents)
			elif self.crossover_algorithm == 'bbwise':
				children = self.bbwise_crossover(parents)

			population = self.mutation(children)

			if (i+1)%10 == 0:
				print(f'{i+1}th generation population: {population}')

		fitness = np.array([self.fitness(indv) for indv in population])
		max_fitness = np.where(fitness == np.max(fitness))
		parameters = population[max_fitness[0][0]]
		return parameters, fitness_history

def setup_logger(name, save_dir, filename="log.txt"):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG) # DEBUG, INFO, ERROR, WARNING
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	stream_handler = logging.StreamHandler(stream=sys.stdout)
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	file_handler = logging.FileHandler(os.path.join(save_dir, filename))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger

def get_timestamp():
	now = datetime.now()
	timestamp = datetime.timestamp(now)
	st = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
	return st

def get_arguments():
	parser = argparse.ArgumentParser(description='4-bit deceptive problem')
	parser.add_argument('--iter', type=str, help='ith iteration', required=True)
	parser.add_argument('--gen', type=int, help='number of generations', default=400)
	parser.add_argument('--pop', type=int, help='population size', default=200)
	parser.add_argument('--indv', type=int, help='individual length', default=100)
	parser.add_argument('--sel', type=str, help='selection algorithm (pwts; rws; rank)', default='pwts')
	parser.add_argument('--c_alg', type=str, help='crossover algorithm (uniform; pointwise; bbwise)', default='pointwise')
	parser.add_argument('--c_prob', type=float, help='crossover probability', default=1.0)
	parser.add_argument('--c_point', type=int, help='crossover points', default=3)
	parser.add_argument('--m_prob', type=float, help='mutation probability', default=0.02)

	args = parser.parse_args()
	return args

def plot_fitness(iter, save_dir, num_generations, fitness_history):
	# visualize how the fitness changes with every generation
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
	fitness_history_max = [np.max(fitness) for fitness in fitness_history]
	plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
	plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')

	plt.legend()
	plt.xlabel('Generations')
	plt.ylabel('Fitness')
	# plt.show()
	plt.savefig(f'{save_dir}/fitness_iter_{iter}')

if __name__=='__main__':
	args = get_arguments()

	logger = setup_logger(name=args.iter, save_dir='logs', filename='{}_fourbit_{}.txt'.format(get_timestamp(), args.iter))
	logger = logging.getLogger(args.iter)
	logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training start!")

	fourbit = Fourbit(crossover_algorithm=args.c_alg,
					  crossover_prob=args.c_prob,
					  crossover_points=args.c_point,
					  mutation_prob=args.m_prob,
					  population_size=args.pop,
					  individual_len=args.indv,
					  num_generations=args.gen,
					  selection_algorithm=args.sel) # rws, pwts, rank

	parameters, fitness_history = fourbit.run()

	delimeter = "   "
	logger.info(
				delimeter.join(
					["Optimal individual: {indv}",
					 "Maximum fitness: {fitness}"]
				 ).format(
					indv=parameters,
					fitness=max(fitness_history[-1]))
			)

	plot_fitness(iter=args.iter, save_dir='figures', num_generations=args.gen, fitness_history=fitness_history)