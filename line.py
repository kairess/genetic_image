import cv2, random, os, sys
import numpy as np
from copy import deepcopy
from skimage.measure import compare_mse
import multiprocessing as mp

filepath = sys.argv[1]
filename, ext = os.path.splitext(os.path.basename(filepath))

img = cv2.imread(filepath)
edges = cv2.Canny(img, threshold1=100, threshold2=150)
edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations = 1)
height, width, channels = img.shape

# hyperparameters
n_initial_genes = 50
n_population = 50
prob_mutation = 0.01
prob_add = 0.3
prob_remove = 0.2

min_length, max_length = 5, 15
save_every_n_iter = 100

# Gene
class Gene():
  def __init__(self):
    self.pt1 = np.array([random.randint(0, width), random.randint(0, height)])

    while True:
      self.pt2 = self.pt1 + np.array([random.randint(-max_length, max_length), random.randint(-max_length, max_length)])
      self.pt2[0] = np.clip(self.pt2[0], 0, width)
      self.pt2[1] = np.clip(self.pt2[1], 0, height)

      dist = np.linalg.norm(self.pt1 - self.pt2)

      if min_length < dist < max_length:
        break

  def mutate(self):
    mutation_size = max(1, int(round(random.gauss(15, 4)))) / 100

    self.pt1[0] = random.randint(
      np.clip(int(self.pt1[0] * (1 - mutation_size)), 0, width),
      np.clip(int(self.pt1[0] * (1 + mutation_size)), 0, width)
    )
    self.pt1[1] = random.randint(
      np.clip(int(self.pt1[1] * (1 - mutation_size)), 0, height),
      np.clip(int(self.pt1[1] * (1 + mutation_size)), 0, height)
    )

    self.pt2[0] = random.randint(
      np.clip(int(self.pt2[0] * (1 - mutation_size)), 0, width),
      np.clip(int(self.pt2[0] * (1 + mutation_size)), 0, width)
    )
    self.pt2[1] = random.randint(
      np.clip(int(self.pt2[1] * (1 - mutation_size)), 0, height),
      np.clip(int(self.pt2[1] * (1 + mutation_size)), 0, height)
    )

# compute fitness
def compute_fitness(genome):
  out = np.zeros((height, width), dtype=np.uint8) * 255

  for gene in genome:
    cv2.line(out, pt1=tuple(gene.pt1), pt2=tuple(gene.pt2), color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

  # mean squared error
  fitness = 255. / compare_mse(edges, out)

  return fitness, out

# compute population
def compute_population(g):
  genome = deepcopy(g)
  # mutation
  if len(genome) < 200:
    for gene in genome:
      if random.uniform(0, 1) < prob_mutation:
        gene.mutate()
  else:
    for gene in random.sample(genome, k=int(len(genome) * prob_mutation)):
      gene.mutate()

  # add gene
  if random.uniform(0, 1) < prob_add:
    genome.append(Gene())

  # remove gene
  if len(genome) > 0 and random.uniform(0, 1) < prob_remove:
    genome.remove(random.choice(genome))

  # compute fitness
  new_fitness, new_out = compute_fitness(genome)

  return new_fitness, genome, new_out

# main
if __name__ == '__main__':
  os.makedirs('result', exist_ok=True)

  p = mp.Pool(mp.cpu_count() - 1)

  # 1st gene
  best_genome = [Gene() for _ in range(n_initial_genes)]

  best_fitness, best_out = compute_fitness(best_genome)

  n_gen = 0

  while True:
    try:
      results = p.map(compute_population, [deepcopy(best_genome)] * n_population)
    except KeyboardInterrupt:
      p.close()
      break

    results.append([best_fitness, best_genome, best_out])

    new_fitnesses, new_genomes, new_outs = zip(*results)

    best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda x: x[0], reverse=True)

    best_fitness, best_genome, best_out = best_result[0]

    # end of generation
    print('Generation #%s, Fitness %s' % (n_gen, best_fitness))
    n_gen += 1

    # visualize
    if n_gen % save_every_n_iter == 0:
      cv2.imwrite('result/%s_%s.jpg' % (filename, n_gen), best_out)

    cv2.imshow('best out', best_out)
    if cv2.waitKey(1) == ord('q'):
     p.close()
     break

  cv2.imshow('best out', best_out)
  cv2.waitKey(0)
