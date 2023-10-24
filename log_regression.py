import numpy as np
import csv
import matplotlib.pyplot as mp

np.random.seed(42)

data = []
with open("data_lr.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader: 
        data.append(row)

data = np.array(data)

coords = data[:, :2]
labels = data[:, 2]

def sigm(t):
    return 1/(1+np.exp(-t))


def pred(coords, weights, bias):
    return (sigm(np.matmul(coords, weights)+bias))


def er_vect(labels, y_pred):
    return [-labels[i]*np.log(y_pred[i]) - (1-labels[i])*np.log(1-y_pred[i]) for i in range(len(labels))]


def mean_er(labels, y_pred):
    temp = er_vect(labels, y_pred)
    return sum(temp)/len(temp)


def dEr(coords, labels, y_pred):
    dErDx1 = [coords[i][0]*(labels[i]-y_pred[i]) for i in range(len(labels))]
    dErDx2 = [coords[i][1]*(labels[i]-y_pred[i]) for i in range(len(labels))]
    dErDb = [labels[i]-y_pred[i] for i in range(len(labels))]

    return dErDx1, dErDx2, dErDb


def gradientDescentStep(coords, labels, weights, bias, step):
    y_pred = pred(coords,weights,bias)
    dErrors = dEr(coords, labels, y_pred)

    weights[0] += sum(dErrors[0])*step
    weights[1] += sum(dErrors[1])*step
    bias += sum(dErrors[2])*step
    e = mean_er(labels, y_pred)
    
    return weights, bias, e


def log_reg(coords, labels, step=0.1, epochs=46):
    weights = np.array(np.random.rand(2, 1))*2 - 1
    b = np.random.rand(1)[0]*2 - 1
    lines = []
    errors = []
    for i in range(epochs):
        coeffs, b, error = gradientDescentStep(coords, labels, weights, b, step)
        lines.append((-coeffs[0]/coeffs[1], -b/coeffs[1]))
        errors.append(error)

    return lines, errors


lines, errors = log_reg(coords, labels)

figure, axis = mp.subplots(1, 2)

axis[0].set_ylim(top=1.25) 
axis[0].set_ylim(bottom=-0.3)
axis[0].set_xlim(left=-0.2)
axis[0].set_xlim(right=1.2)

for i in data:
    if i[2] == 1:
        axis[0].plot(i[0], i[1], color='blue', marker='.', linestyle='none')
    else:
        axis[0].plot(i[0], i[1], color='red', marker='.', linestyle='none')

x = np.linspace(-1, 1, 50)

for i in range(len(lines)):
    if (i % 5 == 0):
        y = lines[i][0]*x+lines[i][1]
        if (i < len(lines)-2):
            axis[0].plot(x, y, label="line{}".format(i), linestyle='dashed')
        else:
            axis[0].plot(x, y, label="line{}".format(i))

axis[0].set_title("Log regression")
axis[0].legend()
axis[1].plot(errors)
axis[1].set_title("Errors")

mp.show()
