from param import sigma_acc, sigma_mia

import matplotlib.pyplot as plt
import seaborn as sns

# Convert dictionary data to lists for plotting
sigmas = list(sigma_acc.keys())
acc_values = list(sigma_acc.values())
mia_values = list(sigma_mia.values())

# Plot main task accuracy
plt.figure(figsize=(10, 5))
sns.lineplot(x=sigmas, y=acc_values, marker="o", label="Main Task Accuracy")
plt.xlabel("Sigma")
plt.ylabel("Accuracy (%)")
plt.title("Main Task Accuracy vs. Sigma")
plt.legend()
plt.grid()
plt.savefig("p100.2f.C_1.gaussian.T_100.sigma_acc.png")
plt.cla()
plt.clf()

# Plot MIA
plt.figure(figsize=(10, 5))
sns.lineplot(x=sigmas, y=mia_values, marker="o", color="orange", label="MIA")
plt.xlabel("Sigma")
plt.ylabel("MIA")
plt.title("MIA vs. Sigma")
plt.legend()
plt.grid()
plt.savefig("p100.2f.C_1.gaussian.T_100.sigma_mia.png")

