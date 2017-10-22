import random
import numpy as np


class NaiveBayes:
    __decisions_distribution = dict()
    __conditional_distribution = dict()

    def __init__(self):
        pass

    def fit(self, data, attributes):
        for probe in data:
            key = probe[len(probe) - 1]
            try:
                self.__decisions_distribution[key] += 1
            except KeyError:
                self.__decisions_distribution[key] = 1
            for i in range(0, len(attributes)):
                if probe[i] == "true":
                    try:
                        self.__conditional_distribution[key, attributes[i]] += 1
                    except KeyError:
                        self.__conditional_distribution[key, attributes[i]] = 1
        for animal_class in self.__decisions_distribution:
            self.__decisions_distribution[animal_class] /= float(len(data))
        for item in self.__conditional_distribution:
            self.__conditional_distribution[item] /= float(len(data))
        pass

    def predict_probe(self, probe, attributes):
        final_conditional_probability = dict()
        for animal_class in self.__decisions_distribution:
            attributes_probability = []
            for i in range(0, len(attributes)):
                if probe[i] == "true":
                    try:
                        attributes_probability.append(self.__conditional_distribution[animal_class, attributes[i]])
                    except KeyError:
                        attributes_probability.append(0)

            final_conditional_probability[animal_class] = np.prod(np.array(attributes_probability)) * self.__decisions_distribution[animal_class]
        return max(final_conditional_probability, key=final_conditional_probability.get)


class BayesUtil:
    def __init__(self):
        pass

    @staticmethod
    def remove_unnecessary_data(data, indexes):
        for row in data:
            i = 0
            for index in indexes:
                del row[index - i]
                i = i + 1
        pass

    @staticmethod
    def remove_unnecessary_attributes(attributes, indexes):
        i = 0
        for index in indexes:
            del attributes[index - i]
            i = i + 1
        for i in range(0, len(attributes)):
            attributes[i] = attributes[i][0]

    pass

    @staticmethod
    def divide_data(data):
        random.shuffle(data)
        teach_size = len(data) * 2 / 3
        return data[:teach_size], data[teach_size:]

    @staticmethod
    def is_classified_properly(probe, selected_class):
        correct = False
        if probe[len(probe)-1] == selected_class:
            correct = True
        return correct
