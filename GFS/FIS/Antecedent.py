from .Term import Term
from .Term import TermAggregate


class Antecedent(object):
    def __init__(self, clause):
        """
        fuzzy rule antecedents
        :param clause: rule antecedents, described using TermAggregate
        """
        self.clause = clause

    def antecedent_terms(self):
        terms = []

        def _find_terms(obj):
            if isinstance(obj, Term):
                terms.append(obj)
            elif obj is None:
                pass
            else:
                assert isinstance(obj, TermAggregate)
                _find_terms(obj.term1)
                _find_terms(obj.term2)

        _find_terms(self.antecedent)
        return terms

    def compute_value(self, input):
        """
        Calculate the value of the rule antecedent based on the input of all fuzzy variables
        :param input: the values ​​of all fuzzy variables, the data structure is a dictionary, the key is the label of the fuzzy variable, and the value is the specific value
        :return: the value of the antecedent of this rule
        """
        return self.clause.compute_value(input)
