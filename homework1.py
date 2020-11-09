import os
import jsonlines
import numpy as np
from ctypes import Union

from confusion_matrix import plot_confusion_matrix
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def load_json_dataset(filename, has_semantic=True):
    """
    :type filename: str
    :type has_semantic: bool
    :rtype: (list, list, set)
    """

    data = list()
    target = list()
    target_list = set()

    with jsonlines.open(filename) as reader:
        for line in reader:
            line_asm = line["lista_asm"]
            data.append(line_asm)

            if has_semantic:
                line_class = line["semantic"]
                target.append(line_class)
                target_list.add(line_class)

    return data, target, target_list


def process_data(data):
    """
    :type data: list
    :rtype: numpy.ndarray
    """

    new_data = list()

    for str_func in data:
        list_func = [instr[1:] for instr in str_func[1:-1].split("', ")]

        xor_count = 0
        for op in list_func:
            xor_count += op.count("xor")

        new_data.append([len(list_func), xor_count])

    return np.array(new_data)


def get_vectorizer(vectorizer_type="count"):
    """
    :type vectorizer_type: str
    :rtype: Union[HashingVectorizer, CountVectorizer, TfidfVectorizer]
    """

    if vectorizer_type == "hashing":
        vectorizer = HashingVectorizer()    # multivariate
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer()      # multinomial
    elif vectorizer_type == "tfid":
        vectorizer = TfidfVectorizer()
    else:
        raise Exception(vectorizer_type + " isn't a valid vectorizer type")

    return vectorizer


def get_model(data_train, target_train, model_type="bernoulli"):
    """
    :type data_train: list
    :type target_train: list
    :type model_type: str
    :rtype: Union[BernoulliNB, MultinomialNB, DecisionTreeClassifier, SVC]
    """

    if model_type == "bernoulli":
        model = BernoulliNB().fit(data_train, target_train)
        print('Bernoulli Model created')
    elif model_type == "multinomial":
        model = MultinomialNB().fit(data_train, target_train)
        print('Multinomial Model created')
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier().fit(data_train, target_train)
    elif model_type == "svm":
        model = SVC(kernel='linear', C=1).fit(data_train, target_train)
    else:
        raise Exception(model_type + " isn't a valid model type")

    return model


def run_blindtest(model, model_type):
    """
    :type model: Union[BernoulliNB, MultinomialNB, DecisionTreeClassifier, SVC]
    :type model_type: str
    :rtype: None
    """

    blindtest_filename: str = "blindtest.json"
    print("Running blindtest: " + blindtest_filename)

    blindtest_data, _, _ = load_json_dataset(blindtest_filename, has_semantic=False)
    X_new = process_data(blindtest_data)

    filename = "results_" + model_type + ".txt"
    full_path = os.path.join("results", filename)

    with open(full_path, "w+") as f:
        for x_new in X_new:
            y_new = model.predict([x_new])
            f.write(y_new[0] + "\n")

    print("Results saved in " + full_path)


if __name__ == "__main__":
    dataset_filename: str = "dataset.json"
    _model_type = "multinomial"

    _data, y_all, class_names = load_json_dataset(dataset_filename)

    X_all = process_data(_data)

    '''
    _vectorizer_type = "tfid"
    _vectorizer = get_vectorizer(_vectorizer_type)
    X_all = _vectorizer.fit_transform(_data)
    '''

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.333, random_state=42)
    print("Train: %d - Test: %d" % (X_train.shape[0], X_test.shape[0]))

    _model = get_model(X_train, y_train, _model_type)
    y_pred = _model.predict(X_test)
    print(classification_report(y_test, y_pred))

    run_blindtest(_model, _model_type)

    plt, _ = plot_confusion_matrix(y_test, y_pred, class_names, normalize=True)

    '''
    f = open(os.path.join("results", "results_" + _vectorizer_type + "_" + _model_type + ".txt"), "w+")
    with jsonlines.open(blindtest_filename) as reader:
        for line in reader:
            X_new = _vectorizer.transform([line["lista_asm"]])
            y_new = _model.predict(X_new)
            f.write(y_new[0] + "\n")

    f.close()
    '''
