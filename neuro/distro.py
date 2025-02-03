import numpy as np
import matplotlib.pyplot as plt

# Gaussian function
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Parameters
mu = 0        # Mean
sigma = 1     # Standard deviation
x = np.linspace(-4, 4, 1000)  # Generate values from -4 to 4

# Compute Gaussian values
y = gaussian(x, mu, sigma)

# Plot
plt.plot(x, y, label=f"Gaussian (μ={mu}, σ={sigma})", color="blue")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Gaussian Distribution")
plt.legend()
plt.grid()
plt.show()
