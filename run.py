import arff

from model import BayesUtil, NaiveBayes

FILE_PATH = "data/zoo.arff"

data_set = arff.load(open(FILE_PATH, 'rb'))
data = data_set['data']
attributes = data_set['attributes']

BayesUtil.remove_unnecessary_data(data, [0, 13])
BayesUtil.remove_unnecessary_attributes(attributes, [0, 13, 14])
teach_data, test_data = BayesUtil.divide_data(data)

naive_bayes = NaiveBayes()
naive_bayes.fit(data, attributes)

selected_class = naive_bayes.predict_probe(test_data[0], attributes)
print "Is classified properly: " + str(BayesUtil.is_classified_properly(test_data[0], selected_class))
