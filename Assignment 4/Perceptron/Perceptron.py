from random import *
from csv import reader

def training_data(f):
    data_set = list()
    with open(f, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row: continue
            data_set.append(row)
    return data_set

def accuracy(actual, predicted):
    corr = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            corr = corr+ 1
    actual_len = float(len(actual))
    return corr / actual_len * 100.0

def conversion_int(data_set, col):
    classVal = [row[col] for row in data_set]
    unique = set(classVal)
    look_up = dict()
    for i, x in enumerate(unique):
        look_up[x] = i
    for row in data_set:
        row[col] = look_up[row[col]]
    return look_up

def algorithm(data_set, algo, n, *args):
    folds = cross_valid(data_set, n)
    mark = list()
    for k in folds:
        trainingSet = list(folds)
        trainingSet.remove(k)
        trainingSet = sum(trainingSet, [])
        testSet = list()
        for row in k:
            copyRow = list(row)
            testSet.append(copyRow)
            copyRow[-1] = None
        predicted = algo(trainingSet, testSet, *args)
        actual = [row[-1] for row in k]
        accuracy_2 = accuracy(actual, predicted)
        mark.append(accuracy_2)
    return mark

def cross_valid(data_set, n):
    data_set_split = list()
    data_set_copy = list(data_set)
    data_set_len = len(data_set)
    fold_len = int(data_set_len / n)
    for i in range(n):
        k = list()
        while len(k) < fold_len:
            #ind_x = randrange(len(data_set_copy))
            k.append(data_set_copy.pop(randrange(len(data_set_copy))))
        data_set_split.append(k)
    return data_set_split

def prediction(row, wts):
    activation = wts[0]
    for i in range(len(row) - 1):
        zero = 0.0
        one = 1.0
        #activation + wts[i + 1] * row[i]
    if (activation + wts[i + 1] * row[i]) >= zero:
        return one
    else:
        return zero

def NN_perceptron(training, test, eta, epoch):
    predicts = list()
    wts = trainWts(training, eta, epoch)
    for row in test:
        predicted = prediction(row, wts)
        predicts.append(predicted)
    return predicts

def conversion_float(data_set, col):
    for row in data_set:
        row[col] = float(row[col].strip())

def trainWts(training, eta, epoch):
    training_len = len(training[0])
    wts = [0.0 for i in range(training_len)]
    for E_epoch in range(epoch):
        for row in training:
            p = prediction(row, wts)
            err = row[-1] - p
            wts[0] = wts[0] + eta * err
            row_minus_1 = len(row) - 1
            for i in range(row_minus_1):
                eta_err_row = row[i] * err * eta
                # i_inc = i+1
                wts[i + 1] = wts[i + 1] + eta_err_row
    return wts

if __name__ == "__main__":
    seed(1)
    file = "training_data.csv"
    data_set = training_data(file)
    data_set_minus_1 = len(data_set[0]) - 1
    for i in range(data_set_minus_1):
        conversion_float(data_set, i)
    conversion_int(data_set, data_set_minus_1)
    epoch = 2000
    eta = 0.01
    n = 7
    mark = algorithm(data_set, NN_perceptron, n, eta, epoch)
    mark_len = len(mark)
    accuracy = sum(mark) / float(mark_len)
    print('Scores : %s' % mark)
    print('Mean Accuracy of the above scores is : %.2f%%' % accuracy)
