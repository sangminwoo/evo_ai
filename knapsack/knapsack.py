import numpy as np
import random
import matplotlib.pyplot as plt

def read_data(filename):
	with open(filename, mode='r') as f:
		weight = []
		value = []
		num_items = 0

		while True:
			item = f.readline()

			if item:
				items = item.split('\t')
				weight.append(int(items[1]))
				value.append(int(items[2][:-1])) # omit '\n'
				num_items += 1
			else:
				break

	return weight, value, num_items

class Knapsack:
	def __init__(self, weight, value, num_items, weight_capacity, crossover_prob, crossover_points, mutation_prob,
				 population_size, num_generations, selection_algorithm):
		self.weight = weight
		self.value = value
		self.num_items = num_items
		self.weight_capacity = weight_capacity
		self.crossover_prob = crossover_prob
		self.crossover_points = crossover_points
		self.mutation_prob = mutation_prob
		self.population_size = population_size
		self.num_generations = num_generations
		self.selection_algorithm = selection_algorithm

	def init_population(self):
		return np.random.randint(0, 2, (2*self.population_size, self.num_items)) # 200x500

	def fitness(self, weight, value, population):
		'''
		fitness = SUM_{i=1}^{n}{ith_gene*ith_value} if SUM_{i=1}^{n}{ith_gene*ith_weight}<=(weight_capacity) else 0
		'''
		fitness = {}
		for i in range(population.shape[0]):
			fit = np.sum(population[i] * value) if np.sum(population[i] * weight) <= self.weight_capacity else 0
			fitness[i] = fit
		return fitness

	def selection_rws(self, fitness, population):
		# Roulette wheel selection
		fitness_values = list(fitness.values())
		probability = np.array(fitness_values) / np.sum(fitness_values)
		pick = np.random.choice(list(fitness.keys()), self.population_size, p=probability)
		return population[pick]

	def selection_pwts(self, fitness, population):
		# Pair-wise tournament selection
		pick = []
		for i in range(0, population.shape[0], 2):
			if fitness[i] > fitness[i+1]:
				pick.append(i)
			else:
				pick.append(i+1)
		return population[pick]

	def crossover(self, parents):
		# 3-point crossover
		children = np.empty((self.population_size*2, self.num_items))
		crossover_points = sorted(random.sample(range(self.num_items), self.crossover_points))

		for i in range(0, self.population_size*2, 2):
			idx = np.random.choice(parents.shape[0], 2, replace=False)

			if random.random() < self.crossover_prob:
				children[i, :crossover_points[0]] = parents[idx[0], :crossover_points[0]]
				children[i, crossover_points[0]:crossover_points[1]] = parents[idx[1], crossover_points[0]:crossover_points[1]]
				children[i, crossover_points[1]:crossover_points[2]] = parents[idx[0], crossover_points[1]:crossover_points[2]]
				children[i, crossover_points[2]:] = parents[idx[1], crossover_points[2]:]

				children[i+1, :crossover_points[0]] = parents[idx[1], :crossover_points[0]]
				children[i+1, crossover_points[0]:crossover_points[1]] = parents[idx[0], crossover_points[0]:crossover_points[1]]
				children[i+1, crossover_points[1]:crossover_points[2]] = parents[idx[1], crossover_points[1]:crossover_points[2]]
				children[i+1, crossover_points[2]:] = parents[idx[0], crossover_points[2]:]
		return children

	def mutation(self, children):
		# bit-wise mutation
		for i in range(children.shape[0]):
			for j in range(children.shape[1]):
				if random.random() < self.mutation_prob:
					children[i, j] = 1 - children[i, j]
		return children

	def run(self):
		fitness_history =[]
		population = self.init_population() # 200x500

		for _ in range(self.num_generations):
			fitness = self.fitness(self.weight, self.value, population) # 200
			fitness_history.append(list(fitness.values()))
			if self.selection_algorithm == 'rws':
				parents = self.selection_rws(fitness, population) # 100x500
			elif self.selection_algorithm == 'pwts':
				parents = self.selection_pwts(fitness, population) # 100x500
			children = self.crossover(parents) # 200x500
			population = self.mutation(children) # 200x500

		fitness_last_gen = self.fitness(self.weight, self.value, population)
		max_fitness = np.where(list(fitness_last_gen.values()) == np.max(list(fitness_last_gen.values())))
		parameters = population[max_fitness[0][0]]
		return parameters, fitness_history

if __name__=='__main__':
	weight, value, num_items = read_data('test.txt')
	knapsack = Knapsack(weight,
						value,
						num_items,
						weight_capacity=13743,
						crossover_prob=0.9,
						crossover_points=3,
						mutation_prob=0.01,
						population_size=100,
						num_generations=100,
						selection_algorithm='rws') # rws, pwts
	parameters, fitness_history = knapsack.run()
	selected_items = np.nonzero(np.arange(num_items) * parameters)[0]
	print(f'Selected items: {selected_items}')

	# visualize how the fitness changes with every generation
	num_generations = 100
	fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
	fitness_history_max = [np.max(fitness) for fitness in fitness_history]
	plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
	plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')
	plt.legend()
	plt.xlabel('Generations')
	plt.ylabel('Fitness')
	plt.show()