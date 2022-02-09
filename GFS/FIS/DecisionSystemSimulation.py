import numpy as np
from .Term import Term
import math


class DecisionSystemSimulation(object):
    def __init__(self, decision_system):
        """
        Simulation using the corresponding FIS decision system
        :param decision_system: decision system
        """
        self.decision_system = decision_system

    def simulation_get_action(self, input_info):
        """
        Carry out fuzzy inference according to the input and the rule base, traverse the entire rule base, find the rule that best matches the current input conditions, obtain the fuzzy degree (Term) of the output fuzzy variable with the highest confidence, and use its index as the decision behavior index.
        :param input_info: the observed value of the input fuzzy variable
        :return: decision behavior index
        """
        rule_lib = self.decision_system.rule_lib # Get the FIS rule library first
        output_fuzzy_variable = self.decision_system.fuzzy_variable_list[-1] # Output fuzzy variables
        action_value = [0 for _ in range(len(output_fuzzy_variable.terms))] # A fuzzy degree (Term) corresponds to an action

        """ Traverse the entire rule base, determine which rule antecedent the current state (observation) most conforms to according to the input, and then calculate which Term of the output fuzzy variable has the highest confidence, and the Term with the highest confidence is the one with the highest confidence behavior"""
        for rule in rule_lib:
            antecedent = rule.antecedent
            consequent = rule.consequent
            strength = antecedent.compute_value(input_info) # Calculate rule strength
            action_id = consequent.clause.id # The id of the Term is the id of the action
            if action_id == -1:
                continue
            """ Multiple rules may correspond to the same action (Term), take the action (Term) Value with the largest confidence (rule strength) as the action value of the behavior """
            action_value[action_id] = max(action_value[action_id], strength)

        return action_value.index(max(action_value)) # Returns the action index of the highest confidence action

    def simulation_get_crisp_value(self, input_info):
        """
        Carry out fuzzy inference according to the input and the rule base, traverse the entire rule base, find the rule that best meets the current input conditions, obtain the fuzzy degree (Term) with the highest confidence in the output fuzzy variable, and defuzzify the Term into a specific numerical output.
        @param input_info: Enter the observed value of the fuzzy variable
        @return: accurate output of consequent after defuzzification
        """
        rule_base = self.decision_system.rule_lib # Get the rule base of FIS first
        output_fuzzy_variable = self.decision_system.fuzzy_variable_list[-1] # Output fuzzy variables
        action_num = len(output_fuzzy_variable.terms)
        output_terms_list = [Term for _ in range(action_num)] # A fuzzy degree (Term) corresponds to an action

        """ Save all terms that output fuzzy variables, one term corresponds to one action """
        for k, v in output_fuzzy_variable.terms.items():
            output_terms_list[v.id] = v

        clip_value = [0 for _ in range(action_num)]

        """ Traverse the entire rule base and save the confidence (rule strength) of all fuzzy degrees (Term) of the output fuzzy variables """
        for rule in rule_base:
            antecedent = rule.antecedent
            consequent = rule.consequent
            strength = antecedent.compute_value(input_info) # Calculate rule strength
            action_id = consequent.clause.id
            if action_id == -1:
                continue
            clip_value[action_id] = max(clip_value[action_id], strength)

        """ Use the clip_value (rule strength) of each Term to interpolate to find the joint graph of all Terms """

        """ Use clip to intercept the triangular membership function, the abscissa of the intersection may be a decimal, but the decimal cannot be calculated when the for loop traverses the abscissa, so you need to manually solve the abscissa of the intersection """
        interp_universe = [] # Use the clip value to truncate the triangle to get the abscissa of the two intersection points
        for term in output_fuzzy_variable.all_term():
            index = term.id
            clip = clip_value[index]
            a = term.trimf[0]
            b = term.trimf[1]
            c = term.trimf[2]
            if clip == 1:
                continue
            if a != b:
                interp_universe.append(clip * (b - a) + a) # Use the similar triangle theorem to find the abscissa of the left intersection: (x-a)/(b-a) = clip_v / 1
            if b != c:
                interp_universe.append(c - clip * (c - b)) # Use the similar triangle theorem to find the abscissa of the right intersection: (c-x)/(c-b) = clip_v / 1

        """ Output the original x-axis value range of each Term of the fuzzy variable """
        normal_universe = [i for i in range(output_fuzzy_variable.universe_down, output_fuzzy_variable.universe_up + 1)]
        """ Incorporate the abscissa of the intersection point into the x-axis integer set of Term, in order to facilitate the later Mean of Max method to calculate the abscissa set of the maximum clip value """
        final_universe = np.union1d(interp_universe, normal_universe) # np.union1d union

        """ Calculate the membership of each Term at each abscissa point (integer point + two intersection points) """
        mf_value = [[] for _ in range(action_num)]
        for index_universe in range(len(final_universe)):
            for index_action in range(action_num):
                mf_value[index_action].append(
                    output_terms_list[index_action].compute_membership_value(final_universe[index_universe]))

        """ Joint maximum membership graph for all Term (maximum y value) """
        output_distribution = []
        for index_universe in range(len(final_universe)):
            Max = 0
            for index_action in range(action_num):
                mf_value[index_action][index_universe] = min(mf_value[index_action][index_universe],
                                                             clip_value[index_action])
                Max = max(Max, mf_value[index_action][index_universe])
            output_distribution.append(Max)

        """ Use mean of maximum to defuzzify the distribution and find the center value of all the abscissa points whose y value is equal to the maximum y value """
        Max = max(output_distribution)
        count = 0
        sum_count = 0
        for i in range(len(output_distribution)):
            if output_distribution[i] == Max:
                sum_count += final_universe[i]
                count += 1

        return sum_count / count
