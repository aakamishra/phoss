import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from phoss.landscaper import NormalLossDecayLandscape

max_time_steps = 100
print('Debugging Landscaper')

landscaper = NormalLossDecayLandscape(
    'phoss/simulator_configs/overfit.json',
    max_time_steps=max_time_steps,
    samples=100
)
sim_loss = landscaper.generate_landscape()
time_range = np.arange(0, max_time_steps)
print(sim_loss[:,0])
plt.plot(time_range, sim_loss, alpha=0.1, color='blue')
plt.plot(time_range, landscaper.true_loss, alpha=0.1, color='red')
plt.show()
plt.close()

sns.heatmap(sim_loss)
plt.show()
plt.close()
