# Above line allows plot to show inline in ipython
import matplotlib.pyplot as plt
# plotting a line
x = [50, 60, 70, 80, 90, 100, 110, 120]
y = [0.8644, 0.8655, 0.8656, 0.8626, 0.8631, 0.8638, 0.8638, 0.8634]
plt.plot(x, y, label="Validation rmse - latent dimension")
plt.xlabel("latent dimension")
plt.ylabel("validation rmse")
# to show title( or say label for multi plot in a figure )
plt.legend()
plt.savefig("q2.png")
