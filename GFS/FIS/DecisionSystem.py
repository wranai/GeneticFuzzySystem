class DecisionSystem(object):
    def __init__(self, fuzzy_variable_list, rule_lib):
        """
        FIS Decision System: Composed of Fuzzy Variable Set and Fuzzy Rule Base
        :param fuzzy_variable_list: fuzzy variable set
        :param rule_lib: Fuzzy rule library
        """
        self.fuzzy_variable_list = fuzzy_variable_list
        self.rule_lib = rule_lib
