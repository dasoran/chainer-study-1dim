import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.1)
y = np.sin(x)
plt.plot(x, y)

plt.savefig('sin.png')

plt.clf()
y = np.cos(x)
plt.plot(x, y)

plt.savefig('cos.png')
