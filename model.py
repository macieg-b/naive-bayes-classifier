import random
import numpy as np


class NaiveBayes:

    def __init__(self):
        self.__decisions_distribution = dict()
        self.__conditional_distribution = dict()
        self.__teach_length = 0
        self.__la_place = False
        pass

    def la_place(self, use):
        if type(use) is bool:
            self.__la_place = use

    def fit(self, teach_data, attributes):
        self.__teach_length = len(teach_data)
        for probe in teach_data:
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
                elif probe[i] == "false":
                    try:
                        self.__conditional_distribution[key, attributes[i]] += 0
                    except KeyError:
                        self.__conditional_distribution[key, attributes[i]] = 0
        pass

    def predict_probe(self, probe, attributes):
        final_conditional_probability = dict()
        if self.__la_place:
            for animal_class in self.__decisions_distribution:
                attributes_probability = []
                for i in range(0, len(attributes)):
                    if probe[i] == "true":
                        probability = (self.__conditional_distribution[animal_class, attributes[i]] + 1) / float(
                            self.__decisions_distribution[animal_class] + 2)
                        attributes_probability.append(probability)
                    elif probe[i] == "false":
                        probability = (self.__conditional_distribution[animal_class, attributes[i]] + 1) / float(
                            self.__decisions_distribution[animal_class] + 2)
                        complement = 1 - probability
                        attributes_probability.append(complement)
                class_probability = (self.__decisions_distribution[animal_class] + 1) / float(
                    self.__teach_length + 2)
                attributes_probability.append(class_probability)
                final_conditional_probability[animal_class] = np.prod(np.array(attributes_probability))
        else:
            for animal_class in self.__decisions_distribution:
                attributes_probability = []
                for i in range(0, len(attributes)):
                    if probe[i] == "true":
                        probability = self.__conditional_distribution[animal_class, attributes[i]] / float(
                            self.__decisions_distribution[animal_class])
                        attributes_probability.append(probability)
                    elif probe[i] == "false":
                        probability = self.__conditional_distribution[animal_class, attributes[i]] / float(
                            self.__decisions_distribution[animal_class])
                        complement = 1 - probability
                        attributes_probability.append(complement)
                class_probability = self.__decisions_distribution[animal_class] / float(self.__teach_length)
                attributes_probability.append(class_probability)
                final_conditional_probability[animal_class] = np.prod(np.array(attributes_probability))

        return max(final_conditional_probability, key=final_conditional_probability.get), final_conditional_probability[
            max(final_conditional_probability, key=final_conditional_probability.get)]


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
    def divide_data(data, ratio):
        random.shuffle(data)
        teach_size = int(len(data) * ratio)
        return data[:teach_size], data[teach_size:]

    @staticmethod
    def is_classified_properly(probe, selected_class):
        correct = False
        if probe[len(probe) - 1] == selected_class:
            correct = True
        return correct

    @staticmethod
    def bayes_error(result, predict_result):
        if len(result) != len(predict_result):
            raise Exception("Arguments length error!")

        correct = 0
        for i in range(0, len(result)):
            if result[i] == predict_result[i]:
                correct += 1
        accuracy = correct / float(len(result))
        error = 1 - accuracy
        return error, accuracy
