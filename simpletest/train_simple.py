#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import computational_graph as c
import chainer.functions as F
from chainer import optimizers


batchsize = 100
n_epoch = 10
n_units = 100

N = 6000

print('generate dataset')
def generate_sample(num, noise_weight): #noise_weight is 1-, 1 is noisiest
  target = np.array([])
  data = np.array([])
  for var in range(0, num):
    x = np.random.rand()
    if x < 0.05:
      y = 0
    elif x < 0.1:
      y = 1
    elif x < 0.15:
      y = 2
    elif x < 0.2:
      y = 3
    elif x < 0.6:
      y = 4
    else:
      y = 5
    x += np.random.randn() / noise_weight
    target = np.append(target, y)
    data = np.append(data, x)
  return [data, target]

curves_data, curves_target = generate_sample(N, 10000)
test_data, test_target = generate_sample(1000, 10)
curves_data = np.append(curves_data, test_data, axis=0)
curves_target = np.append(curves_target, test_target, axis=0)


curves_data = curves_data.reshape((N + 1000, 1))

curves = {}
curves['data'] = curves_data
curves['target'] = curves_target
curves['data'] = curves['data'].astype(np.float32)
curves['target'] = curves['target'].astype(np.int32)

x_train, x_test = np.split(curves['data'],   [N])
y_train, y_test = np.split(curves['target'], [N])
N_test = y_test.size


# Prepare multi-layer perceptron model
model = chainer.FunctionSet(l1=F.Linear(1, 100),
                            l2=F.Linear(100, 6),
                            l3=F.Linear(6,100),
                            l4=F.Linear(100,1000))

def forward(x_data, y_data, train=True):
    # Neural net architecture
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    y = model.l2(h1)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)


l1_W = []
l2_W = []
# Learning loop
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = np.asarray(x_train[perm[i:i + batchsize]])
        y_batch = np.asarray(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_test, batchsize):
        x_batch = np.asarray(x_test[i:i + batchsize])
        y_batch = np.asarray(y_test[i:i + batchsize])

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))


print(model.l1.W.size)
print(model.l2.W.size)

x = np.arange(0, model.l1.W.size, 1)
y = np.array([])
for w in model.l1.W:
  y = np.append(y, w[0])

plt.plot(x, y)
plt.show()

x = np.arange(0,n_units, 1)
y = l2_W[0][len(l2_W[0])-1]

curves_data, curves_target = generate_sample(10, 10000)

curves_data = curves_data.reshape((10, 1))

curves = {}
curves['data'] = curves_data
curves['target'] = curves_target
x_test = curves['data'].astype(np.float32)
y_test = curves['target'].astype(np.int32)

N_test = y_test.size




x = chainer.Variable(np.array(0.3).astype(np.float32).reshape(1,1))
h1 = F.dropout(F.relu(model.l1(x)),  train=False)
y = model.l2(h1)

for i in range(0, 6):
    print(str(i) + ": " + str(F.softmax(y).data[0][i]))


x = np.arange(0, 6, 1)
plt.plot(x, F.softmax(y).data[0])
#plt.show()


from functools import reduce
print(reduce(lambda s, x = x: s + x, F.softmax(y).data[0]))




