### GeneticFuzzySystem Python Package

This project is a GFS/GFT algorithm package implemented with python. Please put the GFS package into the project path when using it, and introduce the toolkit into the project file for use, or use pip install to install the package:

```powershell
pip install gfs
````

(For reference materials on the principle of GFS algorithm, see: [English version](https://www.cs.princeton.edu/courses/archive/fall07/cos436/HIDDEN/Knapp/fuzzy004.htm), [Chinese version](https://blog.csdn.net/qq_38638132/article/details/106477710))

The BaseGFT base class is defined in the GFS library. GFT can support training multiple GFS controllers, one controller used to decide a specific behavior. The following is a brief introduction to the use of the GFT algorithm package. The CartPole-v0 scene in Open-AI gym is used as a training example. For the sample code, see "main.py"

#### 1. How to use the GFT decider

First import the BaseGFT base class from the GFS package,

````python
from GFS.GeneticFuzzySystem import BaseGFT
````

BaseGFT has built-in methods and functions of the basic GFT algorithm, but also reserves abstract methods that need to be implemented by the user (mainly the data interaction process between the algorithm module and the environment), so the user needs to inherit the BaseGFT base class first. On this basis, implement a custom GFT subclass:

````python
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
                "obs1": obs_list[0],
                "obs2": obs_list[1],
                "obs3": obs_list[2],
                "obs4": obs_list[3]
            }

            action = controller.simulation_get_action(obs_input) # Use FIS decider to get action decision
            obs_list, r, done, _ = simulator.step(action)
            fitness += r

            if done:
                break

        return fitness
````

Among them, the start_simulation() method is an abstract method that the user needs to rewrite. This method defines how the GFS algorithm package obtains the score (Fitness/Reward) of a rule base. When the user accesses different simulation environments, how to get the score from the simulation environment Obtaining observations (Observation), scores (Reward), etc. need to be implemented in this method, and this function returns the score value. The example demo uses the CartPole-v0 scene in the gym as an example, and returns the reward of the gym env as the Fitness of the GFS algorithm.

The main function is divided into 4 steps: constructing fuzzy variables, assigning membership functions, constructing rule base objects, and constructing GFS objects. The implementation process is as follows:

````python
if __name__ == '__main__':
    
""" 1. Construct a fuzzy variable, using CartPole-v0 in gym as an example, including 4 observation inputs and 1 behavior output """
    obs1 = FuzzyVariable([-4.9, 4.9], "obs1")
    obs2 = FuzzyVariable([-3.40e+38, 3.40e+38], "obs2")
    obs3 = FuzzyVariable([-0.418, 0.418], "obs3")
    obs4 = FuzzyVariable([-4.18e-01, 4.18e-01], "obs4")

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
    gft = GFT(rule_lib_list=[controller], population_size=40, episode=200, mutation_pro=0.1, cross_pro=0.9,
              simulator=simulator, parallelized=True)
    gft.train()
````

Run the main function (main.py), the schematic diagram of the program is as follows:

<div align=center><img src="assets/GFS.gif" width=800></div>



#### <span id="jump2">2. Save the training model</span>

After each Epoch training is completed, the training model will be stored in the `./models` folder in the entry function directory. The `AllPopulations` folder stores all the individual objects under a population during the entire training process, and saves the current In the training state, the general group state can be restored by loading the population information to continue training; the `OptimalIndividual` folder stores the best individual objects in each generation; the `RuleLibAndMF` folder saves the rule base and affiliation of the optimal individual Function parameters, the file name should be:

````python
[Epoch_N]RuleLib(current_reward)_[No.X].json
or
[Epoch_N]MF(current_reward)_[No.X].json
or
[Epoch_N]Individual(current_reward)_[No.X].json
````

RuleLib represents the rule library storage file, MF represents the membership function parameter storage file, current_reward represents the specific score value of the current individual, [No.X] represents the rule library number (used to confirm which behavior corresponds to the decision), such as:

````python
[Epoch_1]RuleLib(834)_[No.0].json
or
[Epoch_1]MF(834)_[No.0].json
or
[Epoch_1]Individual(834)_[No.X].json
````

The data content in the RuleLib file is as follows:

````python
{
    "RuleLib": [[0, 0, 5], [0, 1, 3], [0, 2, 2], [1, 0, 5], [1, 1, 3], [1, 2, 2], [2, 0, 2], [2, 1, 0], [2, 2, 3]],
    "chromosome": [5, 3, 2, 5, 3, 2, 2, 0, 3]
}
````

The data content in the MF file is as follows:

````python
{
    "mf_offset": [[8, -9, -2, -3, -9, -2, 3, 7, -2, 1, 0, 10, 8, 1, 1, -10, -10, -3 , 6, 6, 9, -2, 2, 8, -9, -4, 3, -9, 4, -1, -1, -7, 10, 4, -8, -6], [-4 , 10, 5, -3, -4, 0, -7, 4, 4, 1, -7, 9, 6, -6, -3, 4, 8, 10, 3, -3, -4, - 4, -8, -5, 5, -1, 9, 6, 3, 7, 10, -2, 6, 3, 10, 4]]
}
````

The data content in the Individual file is as follows:

````python
{"rule_lib_chromosome": ..., "mf_chromosome": ..., "fitness": 247.50138874281646, "flag": 1}
````

The training curve during training is saved in: `./models/tarin_log.png`:

<div align=center><img src="assets/train_log.png" width=500></div>

#### 3. pretrained model loading

When the model is trained, we can call the gft.evalutate() function to view the effect of our training model. The evaluation function needs to enter the save path of the model, such as:

````python
gft.evaluate("models/OptimalIndividuals/[Epoch_47]Individual(144.0).json")
````

After the model is loaded, modify the start_simulation() function and add the render() command to the function to visualize the CartPole-v0 scene:

````python
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

            simulator.render()

            ...
        return fitness
````

Get the following window:

<div align=center><img src="assets/gym_demo.gif" width=500></div>


#### 4. GFTBoard

In order to facilitate the analysis and processing of the training results, you can run `GFS.Tools.GFTBoard.py` to visualize the training results:

1. Fitness graph during training

2. Hyperparameters during training

3. Textual Rule Base for Optimal Models

To use GFTBoard, you need to install the `dearpygui` third-party library, enter the following command to install the third-party library:

```powershell
pip install dearpygui
````

After the installation is complete, run the `GFTBorad.py` file (if you use pip install to download, run it in Terminal: python -m GFS.Tools.GFTBoard) to get the following interface:

<div align=center><img src="assets/GFSBoard1.png" width=600></div>

Click the `Choose Log File` button, select the saved model file, the file is usually saved in the `models` folder, named `events.out.gft` file, after opening, the fitness curve during the training process will appear on the interface , at the same time, the hyperparameter settings during the training process (number of training rounds, gene mutation rate, etc.) appear in the right window, as follows:

<div align=center><img src="assets/GFSBoard2.png" width=600></div>

Next, click the `Choose Individual` button and select an Individual file in the `OptimalIndividuals` folder to view the fuzzy rule base for an optimal model:

<div align=center><img src="assets/GFSBoard3.png" width=600></div>

<div align=center><img src="assets/GFSBoard4.png" width=600></div>
