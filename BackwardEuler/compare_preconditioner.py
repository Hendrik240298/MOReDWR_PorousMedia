import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)

# open results/jacobi.pkl
with open("results/short_jacobi_benchmark.pkl", "rb") as f:
    jacobi = pickle.load(f)

# open results/AMG_SA.pkl
with open("results/short_AMG_SA_benchmark.pkl", "rb") as f:
    AMG_SA = pickle.load(f)

# means jacobi
jacobi["mean_wall_time"] = np.mean(jacobi["wall_time"])
jacobi["mean_iterations"] = np.mean(jacobi["iterations"])

# means AMG_SA
AMG_SA["mean_wall_time"] = np.mean(AMG_SA["wall_time"])
AMG_SA["mean_iterations"] = np.mean(AMG_SA["iterations"])

# state results
print("Jacobi:")
print(f"  mean wall time:  {jacobi['mean_wall_time']}")
print(f"  mean iterations: {jacobi['mean_iterations']}")
print("AMG_SA:")
print(f"  mean wall time:  {AMG_SA['mean_wall_time']}")
print(f"  mean iterations: {AMG_SA['mean_iterations']}")

# plot jacobi["iterations"]
plt.plot(jacobi["wall_time"], label="Jacobi")
plt.plot(AMG_SA["wall_time"], label="AMG_SA")

plt.xlabel("time step")
plt.ylabel("wall time [s]")
plt.legend()
plt.grid()

plt.show()

plt.plot(jacobi["iterations"], label="Jacobi")
plt.plot(AMG_SA["iterations"], label="AMG_SA")

plt.xlabel("time step")
plt.ylabel("#iterations")
plt.legend()
plt.grid()

plt.show()
