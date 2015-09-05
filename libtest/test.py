import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


plt.clf()
x = np.arange(-3, 3, 0.5)
#y = map(lambda x: x + np.random.randn(), np.sin(x))
vfunc = np.vectorize(lambda x: x + np.random.randn()/2)
y = vfunc(np.cos(x))
plt.plot(x, y, ".")

plt.show()

class Data:
 def __init__(self, target, data):
   self.target = target
   self.data = data

curves_target = []
curves_data = []
for var in range(0, 10):
  isSin = True
  if np.random.randn() > 0:
    isSin = False
  x = np.arange(-3, 3, 0.5)
  vfunc = np.vectorize(lambda x: x + np.random.randn()/2)
  if isSin:
    y = vfunc(np.sin(x))
  else:
    y = vfunc(np.cos(x))
  curves_target.append("sin" if isSin else "cos")
  curves_data.append(y)

curves = Data(curves_target, curves_data)

print(curves.data)


