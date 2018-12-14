import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from copy import deepcopy
import time

# Mutation probability : 0-9
# leng_ind = length of an individual


class Population:
	def __init__(self):
		self.path = []
		self.fitness = 0
		self.percent = 0


class Genetic:

	'''
	pop_size: population size
	labyr_size: labyrinth size
	nbr_obstacles: percentage of obstacles with respect to labyrinth area
	leng_ind: length of individual
	mut_prob: probability of mutation
	move: keeps directions 0 to 3: left, up, right, down
	'''
	def __init__(self, pop_size, labyr_size, nbr_obstacles=15, leng_ind=None, mut_prob=3):
		self.pop_percentages = []
		self.pop_size = pop_size
		self.mut_prob = mut_prob
		self.labyr_size = labyr_size
		self.population = []
		self.move = [ [0, -1], [-1, 0], [0, 1], [1, 0]]
		if leng_ind is None:
			self.leng_ind = self.labyr_size * 2
		else:
			self.leng_ind = leng_ind
		print("Leng_ind: %d" % self.leng_ind)

		if nbr_obstacles is None:
			nbr_obstacles = 15
		self.nbr_obstacles = int(self.labyr_size * self.labyr_size * nbr_obstacles / 100)

	def create_labyr(self):
		center = int(self.labyr_size / 2) - 1
		self.labyrinth = [ [0 for i in range(self.labyr_size)] for j in range(self.labyr_size)]
		for i in range(self.nbr_obstacles):
			row = random.randint(0, self.labyr_size-1)
			col = random.randint(0, self.labyr_size-1)
			if ( (row == center and col == center) or 
					(row == 0 and col == 0)):
				continue
			self.labyrinth[row][col] = 1

	def create_population(self):
		self.population = []
		for i in range(self.pop_size):
			pop_tmp = Population()
			for j in range(self.leng_ind):
				pop_tmp.path.append(random.randint(0, 3))
			self.population.append(pop_tmp)


	def genetic(self):
		self.find_fitnesses()
		self.assign_percents()
		times = 5000
		counter = 0
		is_reached = False
		while counter < times:
			new_pop = []
			for i in range(self.pop_size):
				x = self.random_selection()
				y = self.random_selection()
				child = self.reproduce(x, y)
				# print("Child path: {0}".format(child.path))
				if (random.randint(1, 10) <= self.mut_prob):
					# print("Mutating")
					child = self.mutate(child)
				new_pop.append(child)
			self.population = new_pop
			# self.print_population()
			self.find_fitnesses()
			self.assign_percents()
			pop = self.get_best_ind()
			if pop.fitness == self.labyr_size:
				info = "Best Fitnes: "
				info +=  '%2s ' % str(int(pop.fitness))
				info += ' Length of Ind: %2s' % str(len(pop.path))
				info += ' %s %5s %s %s' % (str("Step: "), str(counter), str("/"), str(times))
				info += " -"
				info += ">" * int(counter / 200)
				print(info)
				print("Found!!!!!")
				return pop
			info = "Best Fitnes: "
			info +=  '%2s ' % str(int(pop.fitness))
			info += ' Length of Ind: %2s' % str(len(pop.path))
			info += ' %s %5s %s %s' % (str("Step: "), str(counter), str("/"), str(times))
			info += " -"
			info += ">" * int(counter / 200)
			if (counter % 250 == 0):
				self.plot_labyrint(pop)
			if counter == times - 1 :
				print(info)
			else:
				print(info, end="\r")
			counter += 1


		if counter >= times:
			print("Individual not found")
		return self.get_best_ind()

	'''
		Lengths of individuals may change
	'''
	def genetic_var(self):
		self.find_fitnesses()
		self.assign_percents()
		times = 5000
		counter = 0
		is_reached = False
		while counter < times:
			new_pop = []
			if (self.pop_diverged()):
				# print("Population diverged")
				self.create_population()
				self.find_fitnesses_var()
				self.assign_percents()
			for i in range(self.pop_size):
				x = self.random_selection()
				y = self.random_selection()
				child = self.reproduce_var(x, y)
				# print("Child path: {0}".format(child.path))
				if (random.randint(1, 10) <= self.mut_prob):
					# print("Mutating")
					child = self.mutate(child)
				new_pop.append(child)
			self.population = new_pop
			# self.print_population()
			self.find_fitnesses_var()
			self.assign_percents()
			pop = self.get_best_ind()
			if pop.fitness == self.labyr_size:
				info = "Best Fitnes: "
				info +=  '%2s ' % str(int(pop.fitness))
				info += ' Length of Ind: %2s' % str(len(pop.path))
				info += ' %s %5s %s %s' % (str("Step: "), str(counter), str("/"), str(times))
				info += " -"
				info += ">" * int(counter / 200)
				print(info)
				print("Found!!!!!")
				return pop
			if len(pop.path) > 5 * self.labyr_size:
				self.create_population()
			info = "Best Fitnes: "
			info +=  '%2s ' % str(int(pop.fitness))
			info += ' Length of Ind: %3s' % str(len(pop.path))
			info += '  %s %5s %s %s' % (str("Step: "), str(counter), str("/"), str(times))
			info += " -"
			info += ">" * int(counter / 200)
			if (counter % 100 == 0):
				self.plot_labyrint(pop)
			if counter == times - 1 :
				print(info)
			else:
				print(info, end="\r")
			counter += 1


		if counter >= times:
			print("Individual not found")
		return self.get_best_ind()


	'''
	When using variable length individuals, checks if the length of the individual is 
	shrunk.
	'''
	def pop_diverged(self):
		for ind in self.population:
			if len(ind.path) >= self.labyr_size - 5 :
				return False
		return True

	'''
	Creates an array containing number of individuals.
	An individual takes place in the array with respect to its fitness.
	'''
	def create_pop_percentages(self):
		self.pop_percentages = []
		for individual in self.population:
			for i in range(individual.percent):
				self.pop_percentages.append(individual)
		
	def get_best_ind(self):
		best_ind = self.population[0]
		for i in range(1, self.pop_size):
			if self.population[i].fitness > best_ind.fitness:
				best_ind = self.population[i]
		return best_ind

	def random_selection(self):
		if len(self.pop_percentages) == 0:
			return self.population[random.randint(0, self.pop_size-1)]
		return self.pop_percentages[random.randint(0, len(self.pop_percentages)-1)]

	def reproduce(self, ind1, ind2):
		new_child = Population()
		if (len(ind1.path) <= 1):
			new_child.path = deepcopy(ind2.path)
			return new_child

		loc = random.randint(0, len(ind1.path)-1)

		for i in range(0, loc):
			new_child.path.append(ind1.path[i])

		for i in range(loc, len(ind2.path)):
			new_child.path.append(ind2.path[i])

		return new_child

	def reproduce_var(self, ind1, ind2):
		new_child = Population()
		if (len(ind1.path) <= 1):
			new_child.path = deepcopy(ind2.path)
			return new_child

		loc = random.randint(0, len(ind1.path)-1)

		for i in range(0, loc):
			new_child.path.append(ind1.path[i])

		for i in range(loc, len(ind2.path)):
			new_child.path.append(ind2.path[i])

		for i in range(loc, len(ind1.path)):
			new_child.path.append(ind1.path[i])

		return new_child

	def mutate(self, ind):
		if (len(ind.path) <=  1):
			return ind
		loc = random.randint(0, len(ind.path)-1)
		ind.path[loc] = random.randint(0, 3)
		return ind


	'''
	Fitness of an individual = labyrinth_size - individual's distance from last point to target
	'''
	def find_fitnesses(self):
		blocked = False
		# print("Finding fitnesses")
		# print("Pop size: %d" % len(self.population))
		for individual in self.population:
			blocked = False
			current_point =  [int(self.labyr_size / 2) - 1 for i in range(2)]	
			# print("Current point: {0}".format(current_point))
			for direct in individual.path:
				# print("Direct: {0}".format(direct))
				# print("Move: {0}".format(self.move[direct]))
				tmp = [current_point[0] + self.move[direct][0], 
								current_point[1] + self.move[direct][1]]
				if self.is_blocked(tmp):
					individual.fitness = 0
					blocked = True
					break
				current_point = tmp
			# print("Last point: {0}".format(current_point))
			if not blocked:
				individual.fitness = self.labyr_size - math.sqrt( (0 - current_point[0])**2 
												+ (0 - current_point[1]) ** 2 )
			# print(individual.fitness)

	'''
	Fitness of an individual = labyrinth_size - individual's distance from targe to its closest point
	to target
	'''
	def find_fitnesses_var(self):
		for individual in self.population:
			blocked = False
			current_point =  [int(self.labyr_size / 2) - 1 for i in range(2)]	
			min_dist = self.get_dist(current_point)
			index = 0
			new_path = []
			for i, direct in enumerate(individual.path):
				# print("Direct: {0}".format(direct))
				# print("Move: {0}".format(self.move[direct]))
				tmp = [current_point[0] + self.move[direct][0], 
								current_point[1] + self.move[direct][1]]
				if self.is_blocked(tmp):
					individual.fitness = 0
					blocked = True
					break

				current_point = tmp
				tmp_dist = self.get_dist(current_point)
				if tmp_dist < min_dist:
					min_dist = tmp_dist
					index = i
			# print("Last point: {0}".format(current_point))
			for j in range(index+1):
				new_path.append(individual.path[j])
			# print("Old leng: %d, New leng: %d" % (len(individual.path), len(new_path)))
			individual.path = new_path
			if not blocked:
				individual.fitness = self.labyr_size - self.get_dist(current_point)


	def get_dist(self, point):
		return math.sqrt((0 - point[0])**2 + (0 - point[1])**2)

	def is_blocked(self, location):
		if (location[0] < 0 or location[1] < 0 
				or location[0] >= self.labyr_size or location[1] >= self.labyr_size):
			return True
		if (self.labyrinth[location[0]][location[1]] == 1):
			return True
		return False

	def assign_percents(self):
		# print("Finding percents")
		total = 0
		for individual in self.population:
			total += individual.fitness
		# print("Total: {0}"total)
		for individual in self.population:
			if individual.fitness is not 0 and total is not 0:
				individual.percent = int(1 + individual.fitness  * 100 / total)
			else:
				individual.percent = 0
		total = 0
		for individual in self.population:
			total += individual.percent
		# print("Total percents: {0}".format(total))
		# print("Fitnesses - Percentages: ")
		# for individual in self.population:
		# 	print(individual.fitness, individual.percent)

		self.create_pop_percentages()



	def print_population(self):
		for row in self.population:
			print(row)

	def plot_labyrint1(self, block=False):
		r, c, cat = [], [], []
		# print(self.labyrinth)
		for i, row in enumerate(self.labyrinth):
			for j, col in enumerate(row):
				r.append(i)
				c.append(j)
				cat.append(self.labyrinth[i][j])

		fig, ax = plt.subplots(figsize=(11, 9))
		im = ax.pcolor(self.labyrinth, edgecolors='white', linestyle=':', lw=1)
		for axis in [ax.xaxis, ax.yaxis]:
			axis.set(ticks=np.arange(0.5, self.labyr_size), ticklabels=range(self.labyr_size))
		ax.set_ylim(ax.get_ylim()[::-1])
		ax.xaxis.tick_top()
		ax.yaxis.tick_left()
		title = "Labyr size: " + str(self.labyr_size)
		title += " Pop size: " + str(self.pop_size)
		title += " Mut rate: " + str(self.mut_prob)
		plt.title(title, y=1.08)
		if block is False:
			plt.show(block=False)
			time.sleep(1)
		else:
			plt.show()		
		plt.close('all')
		# r, c, cat = [], [], []
		# for i, row in enumerate(self.labyrinth):
		# 	for j, col in enumerate(row):
		# 		r.append(i)
		# 		c.append(j)
		# 		cat.append(self.labyrinth[i][j])
		# colormap = np.array( ['r', 'g'] )
		# fig = plt.figure(figsize=(12,9))
		# plt.scatter(r, c, s=int(5000/self.labyr_size), marker='s', c=colormap[cat])
		# plt.show()

	def plot_labyrint(self, result, block=False):
		r, c, cat = [], [], []
		# print(self.labyrinth)

		tmp_lab = deepcopy(self.labyrinth)

		current_point =  [int(self.labyr_size / 2) - 1 for i in range(2)]
		tmp_lab[current_point[0]][current_point[1]] = 2
		# print(current_point)
		for i, direct in enumerate(result.path):
			tmp = [current_point[0] + self.move[direct][0], 
								current_point[1] + self.move[direct][1]]
			if self.is_blocked(tmp):
				result.fitness = 0
				blocked = True
				break
			current_point = tmp
			if i == len(result.path) - 1:
				# print("Len: %d" % len(result.path))
				tmp_lab[current_point[0]][current_point[1]] = 3
			else:
				tmp_lab[current_point[0]][current_point[1]] = 2


		for i, row in enumerate(tmp_lab):
			for j, col in enumerate(row):
				r.append(i)
				c.append(j)
				cat.append(tmp_lab[i][j])


		fig, ax = plt.subplots(figsize=(11,9))
		im = ax.pcolor(tmp_lab, edgecolors='white', linestyle=':', lw=1)
		for axis in [ax.xaxis, ax.yaxis]:
			axis.set(ticks=np.arange(0.5, self.labyr_size), ticklabels=range(self.labyr_size))
		ax.set_ylim(ax.get_ylim()[::-1])
		ax.xaxis.tick_top()
		ax.yaxis.tick_left()
		title = "Labyr size: " + str(self.labyr_size)
		title += " Pop size: " + str(self.pop_size)
		title += " Mut rate: " + str(self.mut_prob)
		plt.title(title, y=1.08)
		if block is False:
			plt.show(block=False)
			time.sleep(1)
		else:
			plt.show()
		plt.close('all')


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-ls", "--labyr_size", required=True, type=int)
	ap.add_argument("-mr", "--mutation_rate", required=False, type=int)
	ap.add_argument("-ps", "--pop_size", required=True, type=int)
	ap.add_argument("-li", "--leng_ind", required=False, type=int)
	ap.add_argument("-no", "--nbr_obstacles", required=False, type=int)
	ap.add_argument("-var", "--var_length", required=False, type=int)

	args = vars(ap.parse_args())

	if args["labyr_size"] != None:
		labyr_size = args["labyr_size"]
	else: 
		labyr_size = 10

	if args["mutation_rate"] != None:
		mut_rate = args["mutation_rate"]
	else: 
		mut_rate = 1

	if args["pop_size"] != None:
		pop_size = args["pop_size"]
	else: 
		pop_size = 20

	leng_ind = None
	if args["leng_ind"] != None:
		leng_ind = args["leng_ind"]

	nbr_obstacles = None
	if args["nbr_obstacles"] != None:
		nbr_obstacles = args["nbr_obstacles"]

	var_length = 0
	if args["var_length"] != None:
		var_length = args["var_length"]


	gen = Genetic(pop_size, labyr_size=labyr_size, nbr_obstacles=nbr_obstacles, leng_ind=leng_ind, mut_prob=mut_rate)
	gen.create_labyr()
	gen.create_population()
	gen.find_fitnesses()
	gen.assign_percents()
	gen.plot_labyrint1(True)
	if var_length == 1:
		print("Using variable length of individuals")
		result = gen.genetic_var()
	else:
		print("Using fixed length of individuals")
		result = gen.genetic()
	print("Result: {0}".format(result.path))
	gen.plot_labyrint(result, block=True)


''' 
r = []
c = []
cat = []

for i, row in enumerate(m):
	for j, col in enumerate(row):
		r.append(i)
		c.append(j)
		cat.append(m[i][j])
colormap = np.array( ['r', 'g'] )

plt.scatter(r, c, s=1000, c=colormap[cat])
plt.show()

'''
