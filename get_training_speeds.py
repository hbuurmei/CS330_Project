import numpy as np
import matplotlib.pyplot as plt


losses = np.load('results/dynamics_model_scratch_train_losses.npy')
print(losses.shape)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.savefig('figures/dynamics_model_reptile_adam_train_losses.png')


