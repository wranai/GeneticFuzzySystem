from collections import OrderedDict
from .Term import Term
import numpy as np
import matplotlib.pyplot as plt


class FuzzyVariable(object):
    def __init__(self, universe, label, defuzzify_method='mom'):
        """
        Fuzzy variables: input and output variables used for FIS to make decisions
        :param label: the name of the variable
        :param universe: the value range of the variable
        :param terms: which categories the variable can be fuzzed into
        """
        self.label = label
        self.universe_down = universe[0]
        self.universe_up = universe[1]
        self.terms = OrderedDict()

    def all_term(self):
        all_term = []
        for k, v in self.terms.items():
            all_term.append(v)
        return all_term

    def __getitem__(self, key):
        """
        The "label" term can be accessed using variable["label"]
        :param key: label of term
        :return: the corresponding term
        """
        if key in self.terms.keys():
            return self.terms[key]
        else:
            # Build a pretty list of available mf labels and raise an
            # informative error message
            options = ''
            i0 = len(self.terms) - 1
            i1 = len(self.terms) - 2
            for i, available_key in enumerate(self.terms.keys()):
                if i == i1:
                    options += "'" + str(available_key) + "', or "
                elif i == i0:
                    options += "'" + str(available_key) + "'."
                else:
                    options += "'" + str(available_key) + "'; "
            raise ValueError("Membership function '{0}' does not exist for "
                             "{1} {2}.\n"
                             "Available options: {3}".format(
                                 key, self.__name__, self.label, options))

    def __setitem__(self, key, item):
        """
        A new item can be set for the fuzzy variable using variable["new_label"] = new_term
        :param key: label of term
        :param item: new term instance
        :return:
        """
        if isinstance(item, Term):
            if item.label != key:
                raise ValueError("Term's label must match new key")
            if item.parent is None:
                raise ValueError("Term must not already have a parent")
        else:
            raise ValueError("Unknown Type")
        self.terms[key] = item

    def automf(self, number=5, variable_type='quantity', discrete=False, names=None, special_case=False, special_mf_abc=None, special_case_name="special_case"):
        """
        Generates a specified number of Term, whose membership function defaults to a triangular membership function
        @param discrete: Whether the variable is a discrete variable, for example, a variable representing a categorical type can only take (1, 2, 3), not 1.33
        @param number: The number of terms generated (excluding special membership classes), the default value is 5
        @param variable_type: quality or quantity (quality, quantity)
        @param names: The name of the corresponding Term, its number should be the same as number
        @param special_mf_abc: triangle vertex value of special membership class
        @param special_case: Whether to automatically assign special membership classes
        @param special_case_name: The name of the special subordinate class, the default is special_case
        """
        if not discrete:
            if names is not None:
                # set number based on names passed
                number = len(names)
            else:
                if number not in [3, 5, 7]:
                    raise ValueError("If number is not 3, 5, or 7, "
                                     "you must pass a list of names "
                                     "equal in length to number.")

                if variable_type.lower() == 'quality':
                    names = ['dismal',
                             'poor',
                             'mediocre',
                             'average',
                             'decent',
                             'good',
                             'excellent']
                else:
                    names = ['lowest',
                             'lower',
                             'low',
                             'average',
                             'high',
                             'higher',
                             'highest']

                if number == 3:
                    if variable_type.lower() == 'quality':
                        names = names[1:6:2]
                    else:
                        names = names[2:5]
                if number == 5:
                    names = names[1:6]

            limits = [self.universe_down, self.universe_up]
            universe_range = limits[1] - limits[0]
            widths = [universe_range / ((number - 1) / 2.)] * int(number)
            centers = np.linspace(limits[0], limits[1], number)

            abcs = [[c - w / 2, c, c + w / 2] for c, w in zip(centers, widths)]

            # Clear existing adjectives, if any
            self.terms = OrderedDict()

            """ If there is a special membership class, add the special membership class to the last item """
            if special_case:
                assert special_mf_abc, "[ERROR] You must specify the special_mf_abc value for special mf function."
                names.append(special_case_name)
                abcs.append(special_mf_abc)

            # Repopulate
            index = 0
            for name, abc in zip(names, abcs):
                term = Term(name, self.label, abc, index)
                index +=1
                self[name] = term
        else:
            abcs = [[0, 0, 0] for _ in range(number)]

            index = 0
            for abc in abcs:
                name = str(index)
                term = Term(name, self.label, abc, index)
                index += 1
                self[name] = term

            if special_case:
                self[special_case_name] = Term(special_case_name, self.label, [0, 0, 0], index)

    def show(self):
        color = ['red', 'green', 'blue']
        for k, v in self.terms.items():
            plt.plot(v.trimf, [0, 1, 0], color=color[v.id], label=v.label)
        plt.legend(loc="best")
        plt.xlabel(self.label)
        plt.ylabel("membership")
        plt.show()
