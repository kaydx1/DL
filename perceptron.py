import numpy as np
import matplotlib.pyplot as mp
import csv

np.random.seed(47)

results = []
with open("data_lr.csv") as csvfile:
    # change contents to floats
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:  # each row is a list
        results.append(row)

for i in results:
    if i[2] == 1:
        mp.plot(i[0], i[1], color='blue', marker='.', linestyle='none')
    else:
        mp.plot(i[0], i[1], color='red', marker='.', linestyle='none')

arr = np.array(results)

coords = arr[:, :2]
labels = arr[:, 2]

def step(temp):
    return 1 if temp>=0 else 0


def pred(coords, weights, bias):
    res = (np.matmul(coords, weights)+bias)
    return step(res)


def algorithm(coords, labels, weights, bias, rate):
    # Fill in code
    for i in range(len(coords)):
        predicted = pred(coords[i], weights, bias)
        if labels[i]-predicted == 1:
            weights[0] += rate*coords[i][0]
            weights[1] += rate*coords[i][1]
            bias += rate
        elif labels[i]-predicted == -1:
            weights[0] -= rate*coords[i][0]
            weights[1] -= rate*coords[i][1]
            bias -= rate

    return weights, bias


def train(coords, labels, rate=0.001, epochs=46):

    x_min, x_max = min(i[0] for i in coords), max(i[0] for i in coords)
    y_min, y_max = min(i[1] for i in coords), max(i[1] for i in coords)

    weights = np.array(np.random.rand(2, 1))
    bias = np.random.rand(1)[0] + x_max

    # These are the solution lines that get plotted below.
    coeffs = []
    for i in range(epochs):
        # In each epoch, we apply the perceptron step.
        weights, bias = algorithm(coords, labels, weights, bias, rate)
        coeffs.append((-weights[0]/weights[1], -bias/weights[1]))
    return coeffs


mp.ylim(top=2)  # adjust the top leaving bottom unchanged
mp.ylim(bottom=-2)
mp.xlim(left=-0.5)
mp.xlim(right=1)


lines = train(coords, labels)

coords = np.linspace(-1, 1, 50)

for i in range(len(lines)):
    if (i%5)==0:
        y = lines[i][0]*coords + lines[i][1]
        if (i < len(lines)-2):
            mp.plot(coords, y, label="line{}".format(i), linestyle='dashed')
        else:
            mp.plot(coords, y, label="line{}".format(i))

mp.title("Perceptron")

mp.show()
