import numpy as np

x = np.random.randint(1, 11, 10)
y = np.random.randint(1, 11, 10)
z = np.convolve(x, y)

print(np.convolve([1,2,3], [4, 5, 6]))

import matplotlib.pyplot as plt

plt.subplot(131)
plt.plot(range(len(x)), x)
plt.subplot(132)
plt.plot(range(len(y)), y)
plt.subplot(133)
plt.plot(range(len(z)), z)
plt.show()