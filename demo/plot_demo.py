import matplotlib
import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [0.1, 0.2, 0.3, 0.4, 0.5]

plt.plot(x, y)

ax1 = plt.gca()
ax1.set_title('Title')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
plt.show()