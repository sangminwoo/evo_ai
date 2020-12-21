import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import random
import matplotlib.pyplot as plt

def setup_logger(name, save_dir, filename="log.txt"):
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
	parser = argparse.ArgumentParser(description='Traveling Salesman Problem')
	parser.add_argument('--iter', type=str, help='ith iteration', required=True)
	parser.add_argument('--gen', type=int, help='number of generations', default=500)
	parser.add_argument('--pop', type=int, help='number of populations', default=300)
	parser.add_argument('--sel', type=str, help='selection algorithm (pwts; rws; rank)', default='pwts')
	parser.add_argument('--c_prob', type=float, help='crossover probability', default=0.9)
	parser.add_argument('--c_point', type=int, help='crossover points', default=3)
	parser.add_argument('--m_prob', type=float, help='mutation probability', default=0.01)

	args = parser.parse_args()
	return args

def read_data(filename):
	with open(filename, mode='r') as f:
		where_cities = []

		while True:
			item = f.readline()

			if item:
				coord = []
				items = item.split(' ')
				coord.append(int(items[0]))
				coord.append(int(items[1][:-1])) # omit '\n'
				where_cities.append(coord)
			else:
				break

	return where_cities

def calc_distance(where_cities):
	dist_mat = np.empty((len(where_cities), len(where_cities)))

	for src, (x1, y1) in enumerate(where_cities):
		for trg, (x2, y2) in enumerate(where_cities):
			if src==trg:
				dist_mat[src, trg] = np.inf
			else:
				dist_mat[src, trg] = ((x2-x1)**2+(y2-y1)**2)**0.5

	return dist_mat

def plot_route(where_cities):
	plt.figure(figsize=(20,10))
	x_coords = [x for (x, y) in where_cities]
	y_coords = [y for (x, y) in where_cities]
	plt.scatter(x_coords, y_coords)

	x1 = [x_coords[0], x_coords[-1]]
	y1 = [y_coords[0], y_coords[-1]]
	plt.plot(x_coords, y_coords, 'g', x1, y1, 'g')
	
	for idx in range(len(where_cities)):
		plt.annotate(idx, (x_coords[idx], y_coords[idx]))
	plt.show()

class TravelingSalesman:
	def __init__(self, where_cities, dist_mat, crossover_prob, crossover_points, mutation_prob,
				 population_size, num_generations, selection_algorithm):

		self.where_cities = where_cities
		self.dist_mat = dist_mat
		self.num_cities = len(where_cities)
		self.crossover_prob = crossover_prob
		self.crossover_points = crossover_points
		self.mutation_prob = mutation_prob
		self.population_size = population_size
		self.num_generations = num_generations
		self.selection_algorithm = selection_algorithm

	def init_population(self):
		return np.stack([np.random.choice(self.num_cities, self.num_cities, replace=False) for _ in range(self.population_size*2)]) # 300*2x100

	def fitness(self, population):
		distance = np.zeros((population.shape[0], population.shape[1]))

		for i in range(population.shape[0]):
			for j in range(population.shape[1]):
				if j+1 < population.shape[1]:
					distance[i, j] = self.dist_mat[population[i,j], population[i,j+1]]
				else: # j+1 == population.shape[1]
					distance[i, j] = self.dist_mat[population[i,j], population[i,0]]

		total_dist = np.sum(distance, axis=1)
		fitness = 1/total_dist # -np.log(1/total_dist)
		return fitness, total_dist

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

	def crossover(self, parents):
		# 3-point crossover
		children = np.zeros((parents.shape[0]*2, parents.shape[1]), dtype=int)
		
		for i in range(0, parents.shape[0]*2, 2):
			idx = np.random.choice(parents.shape[0], 2, replace=False)

			if random.random() < self.crossover_prob:
				crossover_points = sorted(np.random.choice(parents.shape[1], self.crossover_points, replace=False))
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
		distance_history = []
		population = self.init_population() # 300*2x100

		for i in range(self.num_generations):
			fitness, total_dist = self.fitness(population) # 300*2
			fitness_history.append(fitness)
			distance_history.append(total_dist)
			if self.selection_algorithm == 'rws':
				parents = self.selection_rws(fitness, population) # 300x100
			elif self.selection_algorithm == 'pwts':
				parents = self.selection_pwts(fitness, population) # 300x100
			elif self.selection_algorithm == 'rank':
				parents = self.selection_rank(fitness, population) # 300x100
			children = self.crossover(parents) # 300*2x100
			population = self.mutation(children) # 300*2x100

			if (i+1)%10 == 0:
				print(f'{i+1}th generation population: {population}')

		fitness, total_dist = self.fitness(population) # 300*2
		max_fitness = np.where(fitness == np.max(fitness))
		parameters = population[max_fitness[0][0]]
		return parameters, fitness_history, distance_history

if __name__=='__main__':
	args = get_arguments()

	logger = setup_logger(name=args.iter, save_dir='logs', filename='{}_salesman_{}.txt'.format(get_timestamp(), args.iter))
	logger = logging.getLogger(args.iter)
	logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training start!")

	where_cities = read_data('tsp_data.txt')
	dist_mat = calc_distance(where_cities)
	salesman = TravelingSalesman(where_cities,
								 dist_mat,
								 crossover_prob=args.c_prob,
								 crossover_points=args.c_point,
								 mutation_prob=args.m_prob,
								 population_size=args.pop,
								 num_generations=args.gen,
								 selection_algorithm=args.sel) # rws, pwts, rank

	parameters, fitness_history, distance_history = salesman.run()
	where_cities = [where_cities[param] for param in parameters]

	delimeter = "   "
	logger.info(
				delimeter.join(
					["Optimal route: {route}",
					 "Optimal distance: {distance})",
					 "Maximum fitness: {fitness}"]
				 ).format(
					route=parameters,
					distance=min(distance_history[-1]),
					fitness=max(fitness_history[-1]))
			)

	# plot_route(where_cities)

	# visualize how the fitness changes with every generation
	# num_generations = 500
	# fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
	# fitness_history_max = [np.max(fitness) for fitness in fitness_history]
	# plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
	# plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')

	# distance_history_mean = [np.mean(distance) for distance in distance_history]
	# distance_history_max = [np.max(distance) for distance in distance_history]
	# plt.plot(list(range(num_generations)), distance_history_mean, label = 'Mean distance')
	# plt.plot(list(range(num_generations)), distance_history_max, label = 'Max distance')

	# plt.legend()
	# plt.xlabel('Generations')
	# plt.ylabel('Fitness')
	# plt.show()