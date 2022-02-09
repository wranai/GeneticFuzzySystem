import json
import random
import copy
from .Antecedent import Antecedent
from .Consequent import Consequent
from .Rule import Rule


class RuleLib(object):
    def __init__(self, fuzzy_variable_list=None):
        self.rule_lib = None
        self.chromosome = None
        if fuzzy_variable_list:
            self.fuzzy_variable_list = copy.deepcopy(fuzzy_variable_list)
            self.generate_random_rule_base(len(self.fuzzy_variable_list) - 1) # Initialize the rule base, the number of default rule preconditions is equal to the total length of fuzzy variables -1
        self.count = 0

    def encode(self, rules):
        """
        Encode the Rule set for representation and storage
        :param rules: Rule collection
        :return:
        """
        rule_lib = []
        chromosome = []
        for rule in rules:
            antecedent_terms = rule.antecedent.clause.antecedent_terms()
            consequent_term = rule.consequent.clause
            rule_code = []
            for term in antecedent_terms:
                rule_code.append(term.id)
            rule_code.append(consequent_term.id)
            chromosome.append(consequent_term.id)
            rule_lib.append(rule_code)
        self.rule_lib = rule_lib
        self.chromosome = chromosome

    def encode_by_chromosome(self, chromosome) -> None:
        """
        Restore the entire rule base by chromosomal genes, from [a1, a2, a3, ...] to [[condition1, a1], [condition2, a2], ...].
        @param chromosome: chromosome group
        @return: None
        """
        rule_code = []
        chromosome_code = []

        length = len(self.fuzzy_variable_list) - 1
        self.count = 0

        def dfs(depth, pre):
            up = len(self.fuzzy_variable_list[depth].all_term())
            for i in range(up):
                now_pre = copy.deepcopy(pre)
                now_pre.append(i)
                if depth != length - 1:
                    dfs(depth + 1, now_pre)
                else:
                    consequent = chromosome[self.count]
                    self.count += 1
                    now_pre.append(consequent)
                    rule_code.append(now_pre)
                    chromosome_code.append(consequent)

        dfs(0, [])
        self.rule_lib = copy.deepcopy(rule_code)
        self.chromosome = copy.deepcopy(chromosome_code)

    def decode(self) -> list:
        """
        The rule base represented by the code is parsed into a Rule object in the FIS system.
        @return: a list of rules containing N Rule objects
        """
        rules = []
        rule_len = len(self.fuzzy_variable_list)
        for rule in self.rule_lib:
            term_list = []
            for index in range(rule_len):
                all_terms = self.fuzzy_variable_list[index].all_term()
                if rule[index] == -1:
                    term_index = len(all_terms) - 1
                else:
                    term_index = rule[index]
                term_list.append(all_terms[term_index])
            clause = term_list[0]
            for i in range(1, (rule_len - 1)):
                clause = clause & term_list[i]
            antecedent_clause = clause
            consequent_clause = term_list[rule_len - 1]
            antecedent = Antecedent(antecedent_clause)
            consequent = Consequent(consequent_clause)
            now_rule = Rule(antecedent, consequent)
            rules.append(now_rule)
        return rules

    def generate_random_rule_base(self, length):
        """
        Fill the rule base by randomly generating rules.
        @param length: how many rule antecedents there are
        @return: None
        """
        rule_code = []
        chromosome_code = []

        def dfs(depth, pre):
            up = len(self.fuzzy_variable_list[depth].all_term())
            for i in range(up):
                now_pre = copy.deepcopy(pre)
                now_pre.append(i)
                if depth != length - 1:
                    dfs(depth + 1, now_pre)
                else:
                    consequent = random.randint(0, len(self.fuzzy_variable_list[depth + 1].all_term())) - 1
                    now_pre.append(consequent)
                    rule_code.append(now_pre)
                    chromosome_code.append(consequent)
        dfs(0, [])
        self.rule_lib = copy.deepcopy(rule_code)
        self.chromosome = copy.deepcopy(chromosome_code)

    def load_rule_base_from_file(self, filepath):
        """
        Load rule base from file
        filepath: rule file storage path
        :return:
        """
        with open(filepath, "r") as f:
            rule_base_dict = json.load(f)
            self.rule_lib = rule_base_dict["RuleLib"]
            self.chromosome = rule_base_dict["chromosome"]

    def save_individual_to_file(self, filepath, individual: dict):
        """
        Save the optimal individual to a local file
        :param individual: the individual object
        :param filepath: path to save file
        :return: None
        """
        with open(filepath, "w") as f:
            json.dump(individual, f)

    def save_rule_base_to_file(self, filepath):
        """
        Save the rulebase to a local file
        :param filepath: path to save file
        :return: None
        """
        with open(filepath, "w") as f:
            rule_base_dict = {'RuleLib': self.rule_lib, 'chromosome': self.chromosome}
            json.dump(rule_base_dict, f)

    def save_mf_to_file(self, filepath, optimal_individual):
        """
        Save membership function parameters to a local file
        @param optimal_individual: optimal individual object
        @param filepath: save file path
        @return: None
        """
        with open(filepath, "w") as f:
            mf_dict = {"mf_offset": optimal_individual["mf_chromosome"]}
            json.dump(mf_dict, f)
