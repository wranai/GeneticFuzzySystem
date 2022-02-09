"""
@Author: P_k_y
@Time: 2020/12/14
"""
import json
import random
import multiprocessing
from .FIS.RuleLib import RuleLib
from .FIS.DecisionSystem import DecisionSystem
from .FIS.DecisionSystemSimulation import DecisionSystemSimulation
import os
import numpy as np
from abc import ABCMeta, abstractmethod
import copy
import time
import matplotlib.pyplot as plt
import pickle


class BaseGFT(metaclass=ABCMeta):
    def __init__(self, rule_lib_list, population_size, episode, mutation_pro, cross_pro, simulator, parallelized=False,
                 protect_elite_num=1, save_interval=1):
        """
        GFT base class.
        @param rule_lib_list: rule library object
        @param population_size: Population size (the number of existing chromosomes, which can be understood as the number of existing rule bases)
        @param episode: how many epochs to train
        @param mutation_pro: mutation probability
        @param cross_pro: cross probability
        @param protect_elite_num: The number of elite chromosomes to be protected, the N chromosomes with the largest fitness do not participate in mutate
        @param simulator: Simulation environment object, used to obtain observations, reports, etc.
        @param save_interval: how many epochs to save the model once
        """
        self.population_size = population_size if not population_size % 2 else population_size + 1
        self.protect_elite_num = protect_elite_num if protect_elite_num <= population_size else population_size
        self.rule_lib_list = rule_lib_list
        self.episode = episode
        self.mutation_pro = mutation_pro
        self.cross_pro = cross_pro
        self.population = []
        self.simulator = simulator
        self.parallelized = parallelized
        self.save_interval = save_interval
        self.fitness_history = {"min_fitness_list": [],
                                "max_fitness_list": [],
                                "average_fitness_list": [],
                                "time_used_list": []}

    def init_population(self) -> None:
        """
        Population initialization function, initialize a specified number of individuals (Individual), the chromosome of an individual is a two-dimensional array,
        Each of these dimensions represents a chromosome of a specific rule base.
        @return: None
        """
        for i in range(self.population_size):
            rule_lib_chromosome = []
            mf_chromosome = []

            fis_num = len(self.rule_lib_list)
            """ Since there are multiple rule bases, different FIS deciders decide different behaviors for different rule bases, so a random chromosome needs to be generated for each rule base. """
            for index in range(fis_num):
                output_fuzzy_variable = self.rule_lib_list[index].fuzzy_variable_list[-1]
                output_terms = output_fuzzy_variable.all_term()

                """ Digitize the term output fuzzy variable by id """
                genes = [term.id for term in output_terms]
                all_term_list = [term for fuzzy_variable in self.rule_lib_list[index].fuzzy_variable_list
                                 for term in fuzzy_variable.all_term()]
                current_rule_lib_chromosome_size = len(self.rule_lib_list[index].rule_lib)

                """ Triangular membership function is used by default, so the chromosome length of membership function is equal to the number of membership functions*3 """
                current_mf_chromosome_size = len(all_term_list) * 3
                current_rule_lib_chromosome = [genes[random.randint(0, len(genes) - 1)] for _ in
                                               range(current_rule_lib_chromosome_size)]
                current_mf_chromosome = [random.randint(-10, 10) for _ in range(current_mf_chromosome_size)]

                """ Add the chromosome representing the current rule base to the chromosome of the individual """
                rule_lib_chromosome.append(current_rule_lib_chromosome)
                mf_chromosome.append(current_mf_chromosome)

            individual = {"rule_lib_chromosome": rule_lib_chromosome, "mf_chromosome": mf_chromosome, "fitness": 0,
                          "flag": 0}

            self.population.append(individual)

    def compute_fitness(self, individual: dict, simulator, individual_id=None, queue=None, min_v=None, max_v=None, average_num=1) -> float:
        """
        To calculate the fitness value of an individual, it is necessary to parse the chromosomes of the individual into a fuzzy rule base to form the FIS in the GFT. The simulation is performed according to the decision maker, and the fitness value is calculated according to the final simulation result. The calculation method can be customized.
        @param individual_id: individual id, used to determine the individual when multi-process computing
        @param queue: Process queue, when the process completes the calculation, add a signal to the queue to facilitate the main process statistics
        @param simulator: simulator object for parallel computing
        @param average_num: Take the experimental results of N experiments as the return value
        @param max_v: To clip the reward, set the maximum value, the default is None, no clip
        @param min_v: To clip the reward, set the minimum value, the default is None, no clip
        @param individual: individual unit
        @return: the fitness value of the individual
        """
        if self.parallelized:
            assert queue, "Parallelized Mode Need multiprocessing.Queue() object, please pass @param: queue."
            assert individual_id is not None, "Parallelized Mode Need individual ID, please pass @param: individual_id."

        gft_controllers = []
        for index, rule_lib in enumerate(self.rule_lib_list):
            """ Parse numerically encoded chromosomes in RuleLib into a list containing multiple Rule objects """
            rl = RuleLib(rule_lib.fuzzy_variable_list)
            rl.encode_by_chromosome(individual["rule_lib_chromosome"][index])
            rules = rl.decode()

            """ Recalculate the membership function parameters of the fuzzy variables, and change the coordinates of the three points of the membership function triangle according to the membership function offset offset in the chromosome. """
            new_fuzzy_variable_list = copy.deepcopy(rule_lib.fuzzy_variable_list)
            count = 0
            for fuzzy_variable in new_fuzzy_variable_list:
                for k, v in fuzzy_variable.terms.items():
                    if (v.trimf[0] == v.trimf[1] and v.trimf[1] == v.trimf[2] and v.trimf[2] == 0) or (
                            v.trimf[0] == -666): # categorical fuzzy variable
                        count += 3
                        continue
                    offset_value = v.span / 20
                    new_a = v.trimf[0] + individual["mf_chromosome"][index][count] * offset_value
                    new_b = v.trimf[1]
                    new_c = v.trimf[2] + individual["mf_chromosome"][index][count + 2] * offset_value
                    new_tri = [new_a, new_b, new_c]
                    new_tri.sort() # Because there are positive and negative translations, point a may move to the right of point b after translation, but the three points of the triangle cannot be out of order, so they need to be sorted from small to large
                    count += 3
                    v.trimf = new_tri

            """ Build FIS Reasoner """
            now_ds = DecisionSystem(new_fuzzy_variable_list, rules)
            now_dss = DecisionSystemSimulation(now_ds)
            gft_controllers.append(now_dss)

        sum_score = 0
        """ Take the average value of N experiments and return the result """
        for i in range(average_num):
            current_reward = self.start_simulation(gft_controllers, simulator)
            current_reward = min(min_v, current_reward) if min_v else current_reward # Minimum single-game simulation clip
            current_reward = max(max_v, current_reward) if max_v else current_reward # Maximum single-game simulation clip
            sum_score += current_reward
        sum_score /= average_num

        if self.parallelized:
            queue.put((individual_id, sum_score))
        else:
            """ Non-parallel can directly assign values ​​to individuals in this function. If parallel processes are used, individual assignments need to be completed in the main process through the return value. """
            individual["fitness"] = sum_score
            individual["flag"] = 1

        return sum_score

    def visualize_progress(self, epoch: int, total_epoch: int, step: int, total_step: int) -> None:
        """
        Visualize the progress bar of the current calculation.
        @param epoch: current epoch round number
        @param total_epoch: the total number of epoch rounds
        @param step: the current number of steps
        @param total_step: total steps
        @return: None
        """
        max_len = 40
        current_progress = int(step / total_step * 40)
        print('\r[Epoch: %d/%d][' % (epoch, total_epoch) + '=' * current_progress + '-' * (
                max_len - current_progress) + ']', end='')

    def save_train_history(self, save_image_path="models/train_log.png", save_log_file_path="models/events.out.gft"):
        """
        Save the training curve to a local file.
        @param save_log_file_path: the path to store the GFT object
        @param save_image_path: the path to save the graph
        @return: None
        """
        """ Save the training graph to the local """
        if save_image_path:
            plt.clf()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title("Training Log For GFS Algorithm")
            plt.xlabel("Epoch(s)")
            plt.ylabel("Fitness")
            plt.plot(self.fitness_history["min_fitness_list"], color='green', alpha=0.8, label='Min Fitness', linestyle='-.')
            plt.plot(self.fitness_history["average_fitness_list"], color='r', alpha=0.8, label='Average Fitness')
            plt.plot(self.fitness_history["max_fitness_list"], color='c', alpha=0.8, label='Max Fitness', linestyle='-.')
            plt.legend()
            plt.savefig(save_image_path)

        """ Save the current object to the local """
        if save_log_file_path:
            log_dict = {
                "fitness_history": self.fitness_history,
                "cross_pro": self.cross_pro,
                "mutate_pro": self.mutation_pro,
                "episode": self.episode,
                "population_size": self.population_size,
                "parallelized": self.parallelized,
                "rule_lib_list": self.rule_lib_list
            }
            f = open(save_log_file_path, 'wb')
            pickle.dump(log_dict, f, 0)
            f.close()

    def compute_with_parallel(self, epoch, total_epoch):
        """
        The fitness value of each individual is calculated in parallel.
        @return: None
        """
        """ Count the individual objects whose flag == 0 (the fitness value has not been calculated) """
        uncalculated_individual_list = list(filter(lambda x: x["flag"] == 0, self.population))
        sum_count, current_count = len(uncalculated_individual_list), 0
        simulator_list = [copy.deepcopy(self.simulator) for _ in range(len(uncalculated_individual_list))]
        process_list, q = [], multiprocessing.Queue()

        """ Create a new process for each chromosome to be calculated """
        for count, individual in enumerate(uncalculated_individual_list):
            process = multiprocessing.Process(target=self.compute_fitness, args=(individual, simulator_list[count], count, q))
            process.start()
            process_list.append(process)

        """ Visual progress bar, exit the function when all chromosome processes have finished computing """
        self.visualize_progress(epoch, total_epoch, current_count, sum_count)
        while True:
            individual_id, fitness = q.get()
            uncalculated_individual_list[individual_id]["fitness"] = fitness
            uncalculated_individual_list[individual_id]["flag"] = 1
            current_count += 1
            self.visualize_progress(epoch, total_epoch, current_count, sum_count)
            if current_count == sum_count:
                break

    def compute_without_parallel(self, epoch, total_epoch):
        """
        The fitness value of each individual is calculated serially.
        @return: None
        """
        sum_count = len(self.population)
        for count, individual in enumerate(self.population):
            count += 1
            self.visualize_progress(epoch, total_epoch, count, sum_count)
            if individual["flag"] == 0:
                self.compute_fitness(individual, self.simulator)

    def select(self, epoch: int, total_epoch: int) -> None:
        """
        Calculate the probability of each chromosome being selected according to fitness, and select whether the chromosome is retained according to the probability.
        @param epoch: the current number of epochs to execute
        @param total_epoch: the total number of epochs to iterate
        @return: None
        """
        start = time.time()

        if self.parallelized:
            self.compute_with_parallel(epoch, total_epoch)
        else:
            self.compute_without_parallel(epoch, total_epoch)

        self.population = sorted(self.population, key=lambda x: x["fitness"])
        fitness_list = [x["fitness"] for x in self.population]
        fitness_list_for_choice = copy.deepcopy(fitness_list)

        """ Move the fitness to the origin, and add 1e-6 to prevent the fitness from being all 0, which will cause an error in the probability calculation later. """
        fitness_list_for_choice = [x - min(fitness_list_for_choice) + 1e-6 for x in fitness_list_for_choice]

        sum_fitness = sum(fitness_list_for_choice)
        fit_pro = [fitness / sum_fitness for fitness in fitness_list_for_choice]

        """ Select population size chromosomes according to probability distribution """
        selected_population = np.random.choice(self.population, self.population_size, replace=False, p=fit_pro)

        use_time = time.time() - start
        max_f, average_f, min_f = max(fitness_list), sum(fitness_list) / len(fitness_list), min(fitness_list)
        self.fitness_history["min_fitness_list"].append(min_f)
        self.fitness_history["average_fitness_list"].append(average_f)
        self.fitness_history["max_fitness_list"].append(max_f)
        self.fitness_history["time_used_list"].append(use_time)

        self.save_train_history()

        print("  Min Fitness: %.2f  |  Max Fitness: %.2f  |  Average Fitness: %.2f  |  Time Used: %.1fs" % (
            min_f, max_f, average_f, use_time))
        self.population = list(selected_population)

    def get_offspring(self, parent: list) -> list:
        """
        After the two parent individuals (Individual) are crossed, two new offspring individuals (Individual) are returned.
        @param parent: parent individual list
        @return: child individual list
        """
        offspring = copy.deepcopy(parent)

        """ Cross-exchange each rule base in the rule base list, but only between the same rule base on different chromosomes, and ensure the same type of rule base through index """
        for index, rule_lib in enumerate(self.rule_lib_list):
            all_term_list = [copy.deepcopy(fuzzy_variable.all_term()) for fuzzy_variable in
                             rule_lib.fuzzy_variable_list]
            current_rule_lib_chromosome_size = len(rule_lib.rule_lib)
            current_mf_chromosome_size = len(all_term_list) * 3

            """ Randomly select the left and right locus index of the swap gene segment (rule base chromosome) """
            cross_left_position_rule_lib = random.randint(0, current_rule_lib_chromosome_size - 1)
            cross_right_position_rule_lib = random.randint(cross_left_position_rule_lib,
                                                           current_rule_lib_chromosome_size - 1)

            """ Swap the gene segment corresponding to the position of the progeny """
            offspring[0]["rule_lib_chromosome"][index][cross_left_position_rule_lib:cross_right_position_rule_lib + 1], \
                offspring[1]["rule_lib_chromosome"][index][cross_left_position_rule_lib:cross_right_position_rule_lib + 1] = \
                offspring[1]["rule_lib_chromosome"][index][
                cross_left_position_rule_lib:cross_right_position_rule_lib + 1], \
                offspring[0]["rule_lib_chromosome"][index][
                cross_left_position_rule_lib:cross_right_position_rule_lib + 1]

            """ Randomly select the left and right locus index of the swap gene segment (membership function chromosome) """
            cross_left_position_mf = random.randint(0, current_mf_chromosome_size - 1)
            cross_right_position_mf = random.randint(cross_left_position_mf, current_mf_chromosome_size - 1)

            offspring[0]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1], \
                offspring[1]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1] = \
                offspring[1]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1], \
                offspring[0]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1]

        """ The new offspring has not been simulated, and the flag is set to 0, which means that the fitness value needs to be calculated later through simulation """
        offspring[0]["flag"] = 0
        offspring[1]["flag"] = 0
        return offspring

    def cross(self) -> None:
        """
        All individuals (Individual) in a population (population) are cross-exchanged according to probability,
        and add offspring to the current population.
        @return: None
        """
        offspring = []
        random.shuffle(self.population)
        """ Crossover between two adjacent individuals """
        d = list(range(0, len(self.population), 2))
        for i in d:
            pro = random.random()
            if pro < self.cross_pro:
                now_offspring = self.get_offspring(self.population[i: i + 2])
                offspring.extend(now_offspring)
        self.population.extend(offspring)

    def mutate(self) -> None:
        """
        The genetic variation function, for each individual in the population (Individual), randomly selects a certain segment of the chromosome,
        Perform a gene mutation for each gene in the segment.
        @return: None
        """
        """ Protects the top N largest chromosomes for fitness from mutation """
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        mutate_population = self.population[self.protect_elite_num:]

        for individual in mutate_population:
            pro = random.random()

            """ Mutate each rulebase """
            if pro < self.mutation_pro:
                for index, rule_lib in enumerate(self.rule_lib_list):
                    output_fuzzy_variable = rule_lib.fuzzy_variable_list[-1]
                    output_terms = output_fuzzy_variable.all_term()
                    genes = [term.id for term in output_terms]
                    all_term_list = [term for fuzzy_variable in self.rule_lib_list[index].fuzzy_variable_list
                                     for term in fuzzy_variable.all_term()]

                    current_rule_lib_chromosome_size = len(rule_lib.rule_lib)
                    current_mf_chromosome_size = len(all_term_list) * 3

                    """ Select the mutation point and mutate all gene segments after this point (rule base) """
                    mutation_pos_rule_lib = random.randint(0, current_rule_lib_chromosome_size - 1)
                    gene_num = len(genes)
                    individual["rule_lib_chromosome"][index][mutation_pos_rule_lib:] = [random.randint(0, gene_num - 1)
                                                                                        for _ in
                                                                                        range(
                                                                                            current_rule_lib_chromosome_size -
                                                                                            mutation_pos_rule_lib)]

                    """ Select the mutation point and mutate all gene segments after this point (membership function) """
                    mutation_pos_mf = random.randint(0, current_mf_chromosome_size - 1)
                    individual["mf_chromosome"][index][mutation_pos_mf:] = [random.randint(-10, 10) for _ in
                                                                            range(
                                                                                current_mf_chromosome_size - mutation_pos_mf)]

                """ The mutated individual flag needs to be reset """
                individual["flag"] = 0

    def get_optimal_individual(self):
        """
        Get the individual object with the highest fitness.
        @return: optimal Individual
        """
        sorted_population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        return sorted_population[0]

    def save_all_population(self, file_path: str):
        """
        Store the entire population in a file (used to save the model checkpoint for the next training session).
        @param file_path: file save directory
        @return: None
        """
        with open(file_path, "w") as f:
            all_population_dict = {"all_population": self.population}
            json.dump(all_population_dict, f)

    def load_all_population(self, file_path):
        """
        Load the entire population from a file.
        @param file_path: The file save directory.
        @return: None
        """
        try:
            with open(file_path, "r") as f:
                all_population_dict = json.load(f)
                self.population = all_population_dict["all_population"]
        except:
            raise IOError("[ERROR] Open File Failed!")

    def save_optimal_individual_to_file(self, path_rule_lib, path_mf, path_individual, optimal_individual):
        """
        Save the individual with the highest score (Individual) into the file.
        @param path_individual: The optimal individual storage directory (excluding the file suffix)
        @param path_rule_lib: rule library file storage directory (excluding file suffix)
        @param path_mf: The directory where the membership function parameter file is stored (excluding the file suffix)
        @param optimal_individual: optimal individual object
        @return: None
        """
        for index, rule_lib in enumerate(self.rule_lib_list):
            """ Put each rule inventory into a different file, the file name contains the rule library number, such as: RuleLib[No.1], which represents the No. 1 rule library """
            current_path_rule_lib = path_rule_lib + "_[No." + str(index) + "].json"
            current_path_mf = path_mf + "_[No." + str(index) + "].json"

            rule_lib.encode_by_chromosome(optimal_individual["rule_lib_chromosome"][index])

            """ Save the rule base, membership functions and individual objects to the local """
            rule_lib.save_rule_base_to_file(current_path_rule_lib)
            rule_lib.save_mf_to_file(current_path_mf, optimal_individual)

            """ Optimal individual object, only need to be stored once """
            if not index:
                current_path_individual = path_individual + ".json"
                rule_lib.save_individual_to_file(current_path_individual, optimal_individual)

    def train(self, save_best_rulelib_mf_path="RuleLibAndMF", save_all_path="AllPopulations",
              save_best_individual_path="OptimalIndividuals", base_path="models", load_last_checkpoint=None) -> None:
        """
        Genetic algorithm training function.
        @param base_path: The total path where the model is stored
        @param save_all_path: kindgroup storage path
        @param save_best_rulelib_mf_path: optimal individual storage path
        @param save_best_individual_path: individual model storage path
        @param load_last_checkpoint: Whether to load the previous model to continue learning, if this parameter is set as the storage path of the all_population.json file
        @return: None
        """

        """ If the directory does not exist, create a new directory folder """
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        save_best_rulelib_mf_path = os.path.join(base_path, save_best_rulelib_mf_path)
        save_all_path = os.path.join(base_path, save_all_path)
        save_best_individual_path = os.path.join(base_path, save_best_individual_path)

        if not os.path.exists(save_best_rulelib_mf_path):
            os.mkdir(save_best_rulelib_mf_path)
        if not os.path.exists(save_best_individual_path):
            os.mkdir(save_best_individual_path)
        if not os.path.exists(save_all_path):
            os.mkdir(save_all_path)

        if not load_last_checkpoint:
            print("\n[INFO]Initializing Rule Lib...")
            self.init_population()
            print("\n[INFO]Finished Initialize Rule Lib, Start to train...\n")
        else:
            print("\n[INFO]Loading last checkpoint: '{}'...".format(load_last_checkpoint))
            self.load_all_population(load_last_checkpoint)
            print("\n[INFO]Finished load Rule Lib, Start to train...\n")

        for count in range(self.episode + 1):
            count += 1
            self.cross()
            self.mutate()
            self.select(count, self.episode)
            optimal_individual = self.get_optimal_individual()

            if not count % self.save_interval:
                self.save_all_population(os.path.join(save_all_path, "all_population{}.json".format(count)))
                self.save_optimal_individual_to_file(
                    os.path.join(save_best_rulelib_mf_path,
                                 "[Epoch_{}]RuleLib({:.1f})".format(count, optimal_individual["fitness"])),
                    os.path.join(save_best_rulelib_mf_path,
                                 "[Epoch_{}]MF({:.1f})".format(count, optimal_individual["fitness"])),
                    os.path.join(save_best_individual_path,
                                 "[Epoch_{}]Individual({:.1f})".format(count, optimal_individual["fitness"])),
                    optimal_individual)

    def evaluate(self, model_name: str):
        """
        Use the stored model to see the training effect.
        @return: None
        """
        individual = json.load(open(model_name, 'r'))

        print("\nLoading Model...")

        gft_controllers = []
        for index, rule_lib in enumerate(self.rule_lib_list):
            """ Parse numerically encoded chromosomes in RuleLib into a list containing multiple Rule objects """
            rl = RuleLib(rule_lib.fuzzy_variable_list)
            rl.encode_by_chromosome(individual["rule_lib_chromosome"][index])
            rules = rl.decode()

            """ Recalculate the membership function parameters of the fuzzy variables, and change the coordinates of the three points of the membership function triangle according to the membership function offset offset in the chromosome. """
            new_fuzzy_variable_list = copy.deepcopy(rule_lib.fuzzy_variable_list)
            count = 0
            for fuzzy_variable in new_fuzzy_variable_list:
                for k, v in fuzzy_variable.terms.items():
                    if (v.trimf[0] == v.trimf[1] and v.trimf[1] == v.trimf[2] and v.trimf[2] == 0) or (
                            v.trimf[0] == -666): # categorical fuzzy variable
                        count += 3
                        continue
                    offset_value = v.span / 20
                    new_a = v.trimf[0] + individual["mf_chromosome"][index][count] * offset_value
                    new_b = v.trimf[1]
                    new_c = v.trimf[2] + individual["mf_chromosome"][index][count + 2] * offset_value
                    new_tri = [new_a, new_b, new_c]
                    new_tri.sort() # Because there are positive and negative translations, point a may move to the right of point b after translation, but the three points of the triangle cannot be out of order, so they need to be sorted from small to large
                    count += 3
                    v.trimf = new_tri

            """ Build FIS Reasoner """
            now_ds = DecisionSystem(new_fuzzy_variable_list, rules)
            now_dss = DecisionSystemSimulation(now_ds)
            gft_controllers.append(now_dss)

        print("\nStart Simulation...")

        self.start_simulation(gft_controllers, self.simulator)

    @abstractmethod
    def start_simulation(self, controllers: list, simulator) -> float:
        """
        Simulator, used to update according to the behavioral decision of the FIS decider, get and return fitness.
        @param simulator: Simulator object.
        @param controllers: contains a list of simulator objects that inherit from DecisionSystemSimulation
        @return: fitness value
        """
        pass


if __name__ == '__main__':
    print("Hello GFS!")
