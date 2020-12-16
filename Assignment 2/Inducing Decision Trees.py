import math
import copy
import random

class TreeNode:
    def __init__(self, split_var=None, branch_left=None, branch_right=None, classifier=None, label=None):
        self.classifier_set = classifier
        self.split_variable = split_var
        self.branch_left = branch_left
        self.branch_right = branch_right
        self.label = label

class Validation:
    def __init__(self, rawData_Val):
        dataList = None
        correct = None
        #self.rawData_Val = self.rawData_Val
        self.accuracy = correct / len(self.raw_data)
        self.attributeNames = dataList[0].split(',')
        self.Attributes = []
        self.raw_data = None

    def load_file(self, f):
        file = open(f, "r")
        data_string = file.read()
        dataList = data_string.splitlines()
        rawData_Val = dataList
        for i in range(0, 20):
            self.Attributes.append(i)
        for index in range(1, len(dataList)):
            rawData_Val[index - 1] = dataList[index].split(',')
        rawData_Val.pop(600)

    def accuracy(self, root):
        if root is None or len(self.rawData_Val) == 0:
            return 0
        correct = 0
        for i in range(len(self.rawData_Val)):
            prediction = self.predicted_value(root, self.rawData_Val[i])
            if prediction == self.rawData_Val[i][20]:
                correct = correct + 1
        return self.accuracy

    def predicted_value(self, root, data_row):
        if root.split_variable == -1:
            return root.label
        if data_row[root.split_variable] == '0':
            return self.predicted_value(root.branch_left, data_row)
        else:
            return self.predicted_value(root.branch_right, data_row)

    def show_accuracy(self):
        print('The accuracy of the decision tree: ' + self.accuracy)

    def Calculate_Accuracy(self, root):
        pass


def impurity(dataset):
    count_class1 = 0
    count_class0 = 0
    total = len(dataset)
    for index in range(0, len(dataset)):
        if dataset[index][20] == '1':
            count_class1 = count_class1 + 1
        if dataset[index][20] == '0':
            count_class0 = count_class0 + 1
    else:
        dTree_impurity = (count_class1 / (total * total)) * count_class0
    return dTree_impurity


def info_impurity_gain(dataset, x):
    # initiate index for x attribute being 0 and 1
    dataset1 = []
    dataset0 = []
    total = len(dataset)
    for index in range(0, len(dataset)):
        if dataset[index][x] == '1':
            dataset1.append(dataset[index])
        if dataset[index][x] == '0':
            dataset0.append(dataset[index])
    if len(dataset0) == 0 or len(dataset1) == 0:
        return 0, dataset0, dataset1
    else:
        parent_impurity = impurity(dataset)
        children_impurity_fraction0 = len(dataset0) / total * impurity(dataset0)
        children_impurity_fraction1 = len(dataset1) / total * impurity(dataset1)
        gain = parent_impurity - (children_impurity_fraction0 + children_impurity_fraction1)
    return gain, dataset0, dataset1


def impurity_split_info(dataset, Attributes):
    max = 0
    dataset1 = []
    dataset0 = []
    split_point = -1
    for attribute in Attributes:
        # if there is 0 gain found, means it cannot be splited ,return split point being -1
        info = info_impurity_gain(dataset, attribute)
        if info[0] != 0 and info[0] > max:
            max = info[0]
            split_point = attribute
            dataset0 = info[1]
            dataset1 = info[2]
    return split_point, dataset0, dataset1


def entropy(dataset):
    count_class1 = 0
    count_class0 = 0
    total = len(dataset)
    # get the number of class 1 and class 0
    for index in range(0, len(dataset)):
        if dataset[index][20] == '1':
            count_class1 = count_class1 + 1
        if dataset[index][20] == '0':
            count_class0 = count_class0 + 1
    if count_class0 == 0:
        entropy0 = 0
    else:
        entropy0 = -count_class0 * math.log2(count_class0 / total) / total
    if count_class1 == 0:
        entropy1 = 0
    else:
        entropy1 = -count_class1 * math.log2(count_class1 / total) / total
    datasetEntropy = entropy1 + entropy0
    return datasetEntropy


def info_gain(dataset, x):
    dataset1 = []
    dataset0 = []
    for index in range(0, len(dataset)):
        if dataset[index][x] == '1':
            dataset1.append(dataset[index])
        if dataset[index][x] == '0':
            dataset0.append(dataset[index])
    if len(dataset0) == 0 or len(dataset1) == 0:
        return 0, dataset0, dataset1
    else:
        gain = entropy(dataset) - (
                    len(dataset0) * entropy(dataset0) / len(dataset) + len(dataset1) * entropy(dataset1) / len(dataset))
        return gain, dataset0, dataset1


def split_info(dataset, Attributes):
    max = 0
    dataset1 = []
    dataset0 = []
    split_point = -1
    for attribute in Attributes:
        # if there is 0 gain found, means it cannot be splited ,return split point being -1
        info = info_gain(dataset, attribute)
        if info[0] != 0:
            if info[0] > max:
                max = info[0]
                split_point = attribute
                dataset0 = info[1]
                dataset1 = info[2]
    return split_point, dataset0, dataset1


def decide_value(dataset):
    # number of class 1 and number of class0
    count_class1 = 0
    count_class0 = 0
    total = len(dataset)
    # get the number of class 1 and class 0
    for index in range(0, total):
        if dataset[index][20] == '1':
            count_class1 += 1
        if dataset[index][20] == '0':
            count_class0 = count_class0 + 1
    if count_class0 > count_class1:
        return '0'
    else:
        return '1'


class DecisionTree:
    def csv_input(self, f):  # csv file input
        file = open(f, "r")
        data_string = file.read()
        dataList = data_string.splitlines()
        raw_data = dataList
        self.Attributes = []
        for i in range(0, 20):
            self.Attributes.append(i)
        self.attributeNames = dataList[0].split(',')
        for index in range(1, len(dataList)):
            raw_data[index - 1] = dataList[index].split(',')
        raw_data.pop(600)
        return raw_data

    def build_impurity_gain(self, dataset):
        self.root = self.split_impurity(dataset, self.Attributes)

    def build(self, dataset):
        self.root = self.split(dataset, self.Attributes)

    def split_impurity(self, dataset, Attributes):
        if len(dataset) == 0:
            return None
        root = TreeNode(-1)
        root.label = decide_value(dataset)
        if len(Attributes) == 0:
            return root
        else:
            splitpoint_info = impurity_split_info(dataset, Attributes)
            splitpoint = splitpoint_info[0]
            if splitpoint == -1:
                return root
            root.split_variable = splitpoint

            newAttributes = []
            for attribute in Attributes:
                if attribute != splitpoint:
                    newAttributes.append(attribute)
            Attributes = newAttributes

            root.branch_left = self.split(splitpoint_info[1], Attributes)
            root.branch_right = self.split(splitpoint_info[2], Attributes)
            return root

    def split(self, dataset, Attributes):
        if len(dataset) == 0:
            return None
        root = TreeNode(-1)
        root.label = decide_value(dataset)
        if len(Attributes) == 0:
            return root
        else:
            splitpoint_info = split_info(dataset, Attributes)
            splitpoint = splitpoint_info[0]
            if splitpoint == -1:
                return root
            root.split_variable = splitpoint
            newAttributes = []
            for attribute in Attributes:
                if attribute != splitpoint:
                    newAttributes.append(attribute)
            Attributes = newAttributes
            root.branch_left = self.split(splitpoint_info[1], Attributes)
            root.branch_right = self.split(splitpoint_info[2], Attributes)

            return root

    def Prunning(self, n, K, datafile):
        prunedTree = self.root
        dtree_validator = Validation()
        dtree_validator.load_file(datafile)

        for i in range(1, n + 1):
            dup_Tree = copy.deepcopy(prunedTree)
            node_list = []
            node_list = self.node_reorder(dup_Tree, node_list)
            nodes_count = len(node_list)
            M = random.randint(1, K)  # M is a random number between 1 and K.

            for j in range(1, M + 1):
                P = random.randint(1, nodes_count)
                randomNode = node_list[P - 1]
                randomNode.split_variable = -1
                randomNode.branch_left = None
                randomNode.branch_right = None

            Accuracy = dtree_validator.Calculate_Accuracy(prunedTree)
            newAccuracy = dtree_validator.Calculate_Accuracy(dup_Tree)
            if newAccuracy >= Accuracy:
                prunedTree = dup_Tree
        self.root = prunedTree
        return prunedTree

    def node_reorder(self, root, node_list):
        node_list.append(root)
        if (root.branch_left is not None) and (root.branch_right is not None):
            self.node_reorder(root.branch_left, node_list)
            self.node_reorder(root.branch_right, node_list)
        return node_list

    def __str__(self):
        return self.ToString(self.root, 0, self.attributeNames)

    def ToString(self, root, level, attributeNames):
        string = ''
        if root.branch_right is None and root.branch_left is None:
            string = string + str(root.label) + '\n'
            return string
        if root is None:
            return ''
        currNode = attributeNames[root.split_variable]
        levelBars = ''
        for i in range(0, level):
            levelBars = levelBars + '| '
        string = string + levelBars
        if root.branch_left.branch_left is None and root.branch_left.branch_right is None:
            string = string + currNode + "= 0 :"
        else:
            string = string + currNode + "= 0 :\n"
        string = string + self.ToString(root.branch_left, level + 1, attributeNames)
        string = string + levelBars
        if root.branch_right.branch_left is None and root.branch_right.branch_right is None:
            string = string + currNode + "= 1 :"
        else:
            string = string + currNode + "= 1 :\n"
        string = string + self.ToString(root.branch_right, level + 1, attributeNames)
        return string

    def __init__(self):
        self.root = TreeNode()
        self.root.split_variable = -1
        self.Attributes = None
        self.attributeNames = None

print("Enter the inputs with spaces: <No. of iterations post-prunning (eg. 10, 15, etc)> <Upper limit of prunning factor (eg. 10, 15, etc)>  <Training_Set directory> <Validation_Set directory> <Test_Set directory> <Print Tree?Yes/No>")

input = input()
input = input.split(' ')
n = int(input[0])
K = int(input[1])
training_set = input[2]
validation_set = input[3]
test_set = input[5]
PrintTree = input[5]
print_PrunnedTree = input[5]

dec_tree_varImp = DecisionTree()
training_data2 = dec_tree_varImp.csv_input(training_set)
dec_tree_varImp.build_impurity_gain(training_data2)

d_tree_ig = DecisionTree()
training_data1 = d_tree_ig.csv_input(training_set)
d_tree_ig.build(training_data1)

if PrintTree != 'yes':
    print("Enter either yes or no to print the tree:")

elif PrintTree == 'yes':
    print('Decision tree D1 (Information Gain Heuristic):')
    print(d_tree_ig)

    print('Decision tree D2 (Variance Impurity Heuristic):')
    print(dec_tree_varImp)

class Prunning:
    pass

if print_PrunnedTree == 'yes':
    print('Tree after Prunning:')
    print(Prunning())

DT_validator = Validation()
DT_validator.load_file(test_set)
print('Accuracy for info gain heuristic is: ')
DT_validator.Calculate_Accuracy(d_tree_ig.root)
print(DT_validator.show_accuracy())

print('Accuracy for variance impurity heuristic is: ')
DT_validator.Calculate_Accuracy(dec_tree_varImp.root)
print(DT_validator.show_accuracy())

d_tree_ig.Prunning(n, K, validation_set)
dec_tree_varImp.Prunning(n, K, validation_set)

print('After prunning, the decision tree accuracy for information gain is:')
DT_validator.Calculate_Accuracy(d_tree_ig.root)
DT_validator.show_accuracy()

print('After prunning, the decision tree accuracy for variance impurity is:')
DT_validator.Calculate_Accuracy(dec_tree_varImp.root)
DT_validator.show_accuracy()
