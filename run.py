import arff

from model import BayesUtil, NaiveBayes

FILE_PATH = "data/zoo.arff"

data_set = arff.load(open(FILE_PATH, 'rb'))
data = data_set['data']

BayesUtil.remove_unnecessary(data, [0, 13])
teach_data, test_data = BayesUtil.divide_data(data)

xu, yu = BayesUtil.split_data_and_classes(teach_data)
naive_bayes = NaiveBayes()
naive_bayes.fit(xu, yu)