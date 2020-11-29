import os
from typing import List

import jsonlines
import numpy as np
from ctypes import Union
from functools import reduce

from pyplot import plot_confusion_matrix, plot_target_distribution
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from networkx import info, json_graph, diameter, strongly_connected_components, DiGraph


def load_json_dataset(filename, has_semantic=True):
    """
    Loads the dataset from a JSON Lines file and parses it to have a list of functions.

    :param filename: A JSON Lines filename
    :type filename: str
    :param has_semantic: True if every entry of the file also has a "semantic" label (target class)
    :type has_semantic: bool
    :returns: The list of functions, the list of target classes and the set of the class names
    :rtype: (list of list, list, set, list of DiGraph)
    """

    data = list()
    target = list()
    target_list = set()
    nx_graphs: List[DiGraph] = list()

    with jsonlines.open(filename) as file:
        for line in file:
            # loads the function as a string from the file
            function: str = line["lista_asm"]
            # parses the function to have a list of its instructions
            instructions: list = [instr[1:] for instr in function[1:-1].split("', ")]
            data.append(instructions)

            nx_graph: DiGraph = json_graph.adjacency_graph(line["cfg"])
            nx_graphs.append(nx_graph)

            # extracts the target classes list
            if has_semantic:
                line_class = line["semantic"]
                target.append(line_class)
                target_list.add(line_class)

    return data, target, target_list, nx_graphs


def process_data(data, nx_graphs):
    """
    :type data: list of list
    :rtype: numpy.ndarray
    """

    x64_ops = {
        "bitwise": ["xor", "or", "and", "not"],
        "arithmetic": ["inc", "dec", "neg", "leaq", "add", "sub", "mul", "divq"],
        "shift": ["sal", "shl", "sar", "shr"],
        "movement": ["lea", "mov", "push", "pop"],
        "calls": ["call", "leave", "ret"]
    }

    new_data = list()

    for i, function in enumerate(data):

        bitwise_count = arithmetic_count = shift_count = calls_count = \
            movement_count = xmm_count = cmp_count = 0

        for instr in function:
            bitwise_count += sum([instr.count(op) for op in x64_ops["bitwise"]])
            shift_count += sum([instr.count(op) for op in x64_ops["shift"]])
            movement_count += sum([instr.count(op) for op in x64_ops["movement"]])
            arithmetic_count += sum([instr.count(op) for op in x64_ops["arithmetic"]])
            calls_count += sum([instr.count(op) for op in x64_ops["calls"]])
            xmm_count += instr.count("xmm")
            cmp_count += instr.count("cmp")

        graph = nx_graphs[i]
        graph_nodes = len(graph.nodes())
        graph_diameter = max([diameter(graph.subgraph(cc_set)) for cc_set in strongly_connected_components(graph)])

        new_data.append([graph_nodes, graph_diameter, movement_count, bitwise_count,
                        xmm_count, arithmetic_count, cmp_count, shift_count, calls_count])

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
    :type data_train: list of list
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
        print('Decision Tree Model created')
    elif model_type == "svm":
        model = SVC(kernel='linear', C=1).fit(data_train, target_train)
        print('SVM Model created')
    elif model_type == "gauss":
        model = GaussianNB().fit(data_train, target_train)
        print('Gaussian Naive Bayes Model created')
    elif model_type == "regression":
        return LogisticRegression().fit(data_train, target_train)
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

    blindtest_data, _, _, nx_graphs = load_json_dataset(blindtest_filename, has_semantic=False)
    X_new = process_data(blindtest_data, nx_graphs)

    filename = "results_" + model_type + ".txt"
    full_path = os.path.join("results", filename)

    with open(full_path, "w+") as f:
        for x_new in X_new:
            y_new = model.predict([x_new])
            f.write(y_new[0] + "\n")

    print("Results saved in " + full_path)


if __name__ == "__main__":
    dataset_filename: str = "dataset.json"  # noduplicatedataset
    _model_type = "decision_tree"

    _data, y_all, class_names, graphs = load_json_dataset(dataset_filename)

    # plot_target_distribution(y_all)

    X_all = process_data(_data, graphs)

    '''
    _vectorizer_type = "tfid"
    _vectorizer = get_vectorizer(_vectorizer_type)
    X_all = _vectorizer.fit_transform([reduce(lambda instr1, instr2 : instr1 + " " + instr2, x) for x in _data])
    '''

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.66, random_state=42)
    print("Train: %d - Test: %d" % (X_train.shape[0], X_test.shape[0]))

    _model = get_model(X_train, y_train, _model_type)
    y_pred = _model.predict(X_test)
    print(classification_report(y_test, y_pred))

    run_blindtest(_model, _model_type)

    plt, _ = plot_confusion_matrix(y_test, y_pred, normalize=True)