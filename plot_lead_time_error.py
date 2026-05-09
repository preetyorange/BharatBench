import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])
    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({'patch_size': self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return config

# 1. Metrics Functions
def compute_rmse(prediction, actual, mean_dims=(1, 2)):
    return np.sqrt(np.mean((prediction - actual) ** 2, axis=mean_dims)).mean()

def compute_mae(prediction, actual, mean_dims=(1, 2)):
    return np.mean(np.abs(prediction - actual), axis=mean_dims).mean()

def compute_acc(prediction, actual, clim):
    pred_anomaly = prediction - clim
    act_anomaly = actual - clim
    pred_norm = pred_anomaly - np.mean(pred_anomaly, axis=(1,2), keepdims=True)
    act_norm = act_anomaly - np.mean(act_anomaly, axis=(1,2), keepdims=True)
    
    num = np.sum(pred_norm * act_norm, axis=(1,2))
    den = np.sqrt(np.sum(pred_norm**2, axis=(1,2)) * np.sum(act_norm**2, axis=(1,2)))
    return np.mean(num / den)

# 2. Data Loading
print("Loading dataset...")
data = xr.open_dataset('dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc')
test_years = slice('2019', '2020')
target_var = 'TMP_prl'

# Using pre-calculated stats for TMP_prl from training
mean_val = data[target_var].sel(time=slice('1990', '2017')).mean().values
std_val = data[target_var].sel(time=slice('1990', '2017')).std().values

subset = data[target_var].sel(time=test_years).values
clim_val = data[target_var].sel(time=slice('1990', '2017')).mean('time').values

norm_data = (subset - mean_val) / std_val

# 3. Model Loading
print("Loading Vanilla ViT...")
custom_objects = {'Patches': Patches, 'PatchEncoder': PatchEncoder}
model = keras.models.load_model('IMDAA_Transformer_T850_5days.keras', custom_objects=custom_objects)

# 4. Autoregressive Inference
lead_times = [5, 10, 15]
steps = [20, 40, 60] # 5 days * 4 (6-hourly) = 20 steps
max_step = steps[-1]

print("Preparing test sequences...")
X_0 = norm_data[:-max_step, ..., None] # Shape: (N, 32, 32, 1)

Y_actuals = {}
for i, step in enumerate(steps):
    Y_actuals[step] = subset[step: len(subset) - max_step + step]

metrics = {'rmse': [], 'mae': [], 'acc': []}

print("Running Autoregressive Inference...")
# Day 5 Prediction
pred_5_norm = model.predict(X_0, batch_size=64)
pred_5 = pred_5_norm.squeeze() * std_val + mean_val

metrics['rmse'].append(compute_rmse(pred_5, Y_actuals[20]))
metrics['mae'].append(compute_mae(pred_5, Y_actuals[20]))
metrics['acc'].append(compute_acc(pred_5, Y_actuals[20], clim_val))

# Day 10 Prediction (Feed Day 5 back into model)
pred_10_norm = model.predict(pred_5_norm, batch_size=64)
pred_10 = pred_10_norm.squeeze() * std_val + mean_val

metrics['rmse'].append(compute_rmse(pred_10, Y_actuals[40]))
metrics['mae'].append(compute_mae(pred_10, Y_actuals[40]))
metrics['acc'].append(compute_acc(pred_10, Y_actuals[40], clim_val))

# Day 15 Prediction (Feed Day 10 back into model)
pred_15_norm = model.predict(pred_10_norm, batch_size=64)
pred_15 = pred_15_norm.squeeze() * std_val + mean_val

metrics['rmse'].append(compute_rmse(pred_15, Y_actuals[60]))
metrics['mae'].append(compute_mae(pred_15, Y_actuals[60]))
metrics['acc'].append(compute_acc(pred_15, Y_actuals[60], clim_val))

# 5. Plotting (Replicating Paper Style)
print("Generating Plot...")
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel('Lead Days', fontweight='bold')
ax1.set_ylabel('Error (K)', color='black', fontweight='bold', fontsize=12)

# Plot RMSE and MAE on primary Y axis
l1, = ax1.plot(lead_times, metrics['rmse'], color='blue', label='RMSE')
l2, = ax1.plot(lead_times, metrics['mae'], color='green', label='MAE')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(lead_times)

# Create secondary Y axis for ACC
ax2 = ax1.twinx()
ax2.set_ylabel('ACC', color='red', fontweight='bold', fontsize=12)
l3, = ax2.plot(lead_times, metrics['acc'], color='red', label='ACC')
ax2.tick_params(axis='y', labelcolor='red')

# Create unified legend
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Vanilla ViT Autoregressive Error over Lead Days (T850)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('lead_time_error.png', dpi=300)
print("Saved lead_time_error.png")
