class Rule(object):
    def __init__(self, antecedent, consequent):
        """
        Fuzzy rule definition, each fuzzy rule consists of rule antecedents and rule results
        :param antecedent: rule antecedent
        :param consequent: rule result
        """
        self.antecedent = antecedent
        self.consequent = consequent

    def get_rule_str(self):
        antecedent_string = self.antecedent.clause.get_string()
        consequent_string = self.consequent.clause.parent + ' is ' + self.consequent.clause.label
        rule_string = antecedent_string + ' THEN ' + consequent_string
        return rule_string
