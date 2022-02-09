class Consequent(object):
    def __init__(self, clause):
        """
        Fuzzy rule results, described using TermAggregate
        :param clause: rule result (usually represented by a Term)
        """
        self.clause = clause
