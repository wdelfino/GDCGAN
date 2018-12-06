import numpy as np

data = np.array((0.1978, 0.1015, 0.1236, 0.1193, 0.1591, 0.1102, 0.1733, 0.1588))

print(np.mean(data))

import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4, 5, 6, 7, 8], [0.1978, 0.1015, 0.1236, 0.1193, 0.1591, 0.1102, 0.1733, 0.1588], 'ro')
plt.axis([0, 10, 0, 0.3])
plt.show()