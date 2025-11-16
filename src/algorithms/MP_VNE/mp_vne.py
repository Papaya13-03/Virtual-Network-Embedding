import random
import math

from algorithms.MP_VNE.global_controller import GlobalController
from types.substrate import SubstrateNetwork

class MP_VNE:
    def __init__(self, snetwork: SubstrateNetwork):
        self.global_controller = GlobalController(snetwork)

    def handle_mapping_request(self, request):
        candidate_nodes = self.global_controller.process_request(request)

        best_mapping = self.pso(candidate_nodes)

        self.map(best_mapping)

    def pso(self, candidates):
        num_of_particles = 50
        num_of_iteration = 30
        num_of_vnode = len(candidates)

        population = []
        for _ in range(num_of_particles):
            particle = [random.choice(candidates[i]) for i in range(num_of_vnode)]
            population.append(particle)

        velocities = [[0 for _ in range(num_of_vnode)] for _ in range(num_of_particles)]

        pbest = population[:]
        pbest_score = [self.fitness(p) for p in population]
        gbest = pbest[pbest_score.index(min(pbest_score))]
        gbest_score = min(pbest_score)

        w = 0.7
        c1 = 1.5
        c2 = 1.5

        for _ in range(num_of_iteration):
            for i in range(num_of_particles):
                for j in range(num_of_vnode):
                    r1 = random.random()
                    r2 = random.random()
                    velocities[i][j] = (w * velocities[i][j] +
                                        c1 * r1 * (pbest[i][j] - population[i][j]) +
                                        c2 * r2 * (gbest[j] - population[i][j]))

                    new_pos = population[i][j] + velocities[i][j]

                    idx = int(abs(new_pos) % len(candidates[j]))
                    population[i][j] = candidates[j][idx]

                if random.random() < 0.1:
                    mutate_idx = random.randint(0, num_of_vnode - 1)
                    population[i][mutate_idx] = random.choice(candidates[mutate_idx])

                score = self.fitness(population[i])
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score

            current_best_score = min(pbest_score)
            if current_best_score < gbest_score:
                gbest = pbest[pbest_score.index(current_best_score)][:]
                gbest_score = current_best_score

        return gbest

    def fitness(self, mapping):
        node_cost = sum(node.cpu_cost for node in mapping)
        link_cost = 0
        for i in range(len(mapping) - 1):
            link_cost += self.global_controller.link_cost(mapping[i], mapping[i+1])
        return node_cost + link_cost

    def map(self, best_mapping):
        self.global_controller.commit_mapping(best_mapping)

    def unmap(self):
        self.global_controller.release_resources()