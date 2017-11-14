import arff
import matplotlib.pyplot as plt

from model import BayesUtil, NaiveBayes
import numpy as np

FILE_PATH = "data/zoo.arff"
REPETITION = 50
data_set = arff.load(open(FILE_PATH, 'rb'))
data = data_set['data']
attributes = data_set['attributes']

BayesUtil.remove_unnecessary_data(data, [0, 13])
BayesUtil.remove_unnecessary_attributes(attributes, [0, 13, 14])

x_vector = range(1, 100)
accuracy_array = np.zeros(len(x_vector))
accuracy_array_laplace = np.zeros(len(x_vector))

for i in range(0, REPETITION):
    print "Epoch %s" % str(i+1)
    for j in x_vector:
        alfa = j / float(100)
        teach_data, test_data = BayesUtil.divide_data(data, alfa)

        naive_bayes = NaiveBayes()
        naive_bayes.la_place(False)
        naive_bayes.fit(teach_data, attributes)

        naive_bayes_laplace = NaiveBayes()
        naive_bayes_laplace.la_place(True)
        naive_bayes_laplace.fit(teach_data, attributes)

        classes = []
        predicted_classes = []

        classes_laplace = []
        predicted_classes_laplace = []
        for probe in test_data:
            selected_class, probability = naive_bayes.predict_probe(probe, attributes)
            classes.append(probe[len(probe) - 1])
            predicted_classes.append(selected_class)

            selected_class_laplace, probability_laplace = naive_bayes_laplace.predict_probe(probe, attributes)
            classes_laplace.append(probe[len(probe) - 1])
            predicted_classes_laplace.append(selected_class_laplace)

        error, accuracy = BayesUtil.bayes_error(classes, predicted_classes)
        error_laplace, accuracy_laplace = BayesUtil.bayes_error(classes_laplace, predicted_classes_laplace)

        if j-1 == 0:
            a = "test"
        accuracy_array[j - 1] += accuracy
        accuracy_array_laplace[j - 1] += accuracy_laplace

accuracy_array /= float(REPETITION)
accuracy_array_laplace /= float(REPETITION)

plt.figure()
plt.title("Classification accuracy")
plt.xlabel("Teaching probes amount - alfa")
plt.ylabel("Accuracy")
plt.plot(x_vector, accuracy_array, '#ff796c')
plt.plot(x_vector, accuracy_array_laplace, '#0165fc')
plt.legend(('Naive', 'La Place'),
           loc='center right')
plt.axis([0, 100, 0, 1.05])
plt.show()
