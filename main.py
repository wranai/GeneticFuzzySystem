"""
@Author: P_k_y
@Time: 2020/12/16
"""
from GFS.FIS.FuzzyVariable import FuzzyVariable
from GFS.FIS.RuleLib import RuleLib
from GFS.GeneticFuzzySystem import BaseGFT
import gym


class GFT(BaseGFT):

    def __init__(self, rule_lib_list, population_size, episode, mutation_pro, cross_pro, simulator, parallelized):
        """
        Implement custom GFT subclasses (inherited from BaseGFT base class) and implement custom computational simulation methods.
        @param rule_lib_list: rule library object
        @param population_size: Population size (the number of existing chromosomes, which can be understood as the number of existing rule bases)
        @param episode: how many epochs to train
        @param mutation_pro: mutation probability
        @param cross_pro: cross probability
        @param simulator: Simulator object, used to get observations and reports
        @param parallelized: whether to enable multi-process parallel computing
        """
        super().__init__(rule_lib_list, population_size, episode, mutation_pro, cross_pro, simulator, parallelized)

    """ Implement parent class abstract method """
    def start_simulation(self, controllers: list, simulator) -> float:
        """
        Customize the data interaction process between the GFT algorithm module and the simulator Simulator (gym), and return the reward value of the simulator.
        @param simulator: Simulator object
        @param controllers: List of controllers, one controller decides one action.
        @return: fitness
        """
        controller = controllers[0]
        fitness = 0

        obs_list = simulator.reset()
        for _ in range(1000):

            # simulator.render()

            """ CartPole-v0 contains a total of 4 observations, which need to be split into 4 fuzzy variable inputs in the FIS decider. """
            obs_input = {
                "car_pos": obs_list[0],
                "car_speed": obs_list[1],
                "pole_angle": obs_list[2],
                "pole_speed": obs_list[3]
            }

            action = controller.simulation_get_action(obs_input) # Use FIS decider to get action decision
            obs_list, r, done, _ = simulator.step(action)
            fitness += r

            """ Reward Shaping: The smaller the angle between the pole and the vertical plane, the higher the score """
            # angle = abs(obs_list[2])
            # r_shaping = (0.418 - angle) / 0.418
            #
            # fitness += r_shaping

            if done:
                break

        return fitness


def create_gft(simulator) -> GFT:
    """
    Establish GFT objects, and establish fuzzy variables and rule bases according to specific scenarios.
    @return: GFT object
    """

    """ 1. Construct a fuzzy variable, using CartPole-v0 in gym as an example, containing a total of 4 observation inputs and 1 behavior output """
    obs1 = FuzzyVariable([-4.9, 4.9], "car_pos")
    obs2 = FuzzyVariable([-3.40e+38, 3.40e+38], "car_speed")
    obs3 = FuzzyVariable([-0.418, 0.418], "pole_angle")
    obs4 = FuzzyVariable([-4.18e-01, 4.18e-01], "pole_speed")

    action = FuzzyVariable([0, 1], "action")

    """ 2. Assign membership functions to fuzzy variables """
    obs1.automf(5)
    obs2.automf(5)
    obs3.automf(5)
    obs4.automf(5)
    action.automf(2, discrete=True) # Action output is a discrete fuzzy variable

    """ 3. Build RuleLib rule base """
    controller = RuleLib([obs1, obs2, obs3, obs4, action])

    """ 4. Build the GFT object """
    return GFT(rule_lib_list=[controller], population_size=20, episode=200, mutation_pro=0.1, cross_pro=0.9,
               simulator=simulator, parallelized=False)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    gft = create_gft(env)
    gft.train()
    # gft.evaluate("models/OptimalIndividuals/[Epoch_24]Individual(151.0).json")
