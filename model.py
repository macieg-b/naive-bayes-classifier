import random


class NaiveBayes:
    __decisions_distribution = dict()
    __conditional_distribution = ""

    def __init__(self):
        pass

    def fit(self, xu, yu):
        animal_classes = dict()
        for animal_class in set(yu):
            animal_classes[animal_class] = 0

        for i in range(0, len(xu)):
            animal_classes[yu[i]] += 1

        for animal_class in set(yu):
            self.__decisions_distribution[animal_class] = float(animal_classes[animal_class]) / len(yu)

        pass

    def predict_probe(self, x):
        return


class BayesUtil:
    def __init__(self):
        pass

    @staticmethod
    def remove_unnecessary(data, indexes):
        for row in data:
            i = 0
            for index in indexes:
                del row[index - i]
                i = i + 1
        pass

    @staticmethod
    def divide_data(data):
        random.shuffle(data)
        teach_size = len(data) * 2 / 3
        return data[:teach_size], data[teach_size:]

    @staticmethod
    def split_data_and_classes(data):
        xu = []
        yu = []
        for row in data:
            xu.append(row[:len(row) - 1])
            yu.append(row[len(row) - 1])
        return xu, yu

    @staticmethod
    def bayes_error(y, pred_y):
        pass
