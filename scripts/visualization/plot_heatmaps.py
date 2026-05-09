import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. Custom Layers from MultiVar Transformer
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

# 2. Load Data
print("Loading dataset...")
data = xr.open_dataset('dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc')

train_years = slice('1990', '2017')
test_years = slice('2019', '2020')
lead_time_steps = 20

variables = ['HGT_prl', 'TMP_prl', 'TMP_2m', 'APCP_sfc']

# Compute standardization parameters based ONLY on training data
mean_dict = {}
std_dict = {}
for var in variables:
    train_data = data[var].sel(time=train_years)
    mean_dict[var] = train_data.mean()
    std_dict[var] = train_data.std()

def prepare_data(data, target_years, variables, mean_dict, std_dict, lead_steps):
    X_channels = []
    subset_data = data.sel(time=target_years)
    for var in variables:
        norm_var = (subset_data[var] - mean_dict[var]) / std_dict[var]
        X_ch = norm_var.isel(time=slice(None, -lead_steps)).values
        X_channels.append(X_ch)
    X = np.stack(X_channels, axis=-1)
    
    target_var = 'TMP_prl'
    norm_target = (subset_data[target_var] - mean_dict[target_var]) / std_dict[target_var]
    Y = norm_target.isel(time=slice(lead_steps, None)).values[..., None]
    
    target_ds = data[target_var].sel(time=target_years)
    return X, Y, target_ds

X_test, _, target_ds = prepare_data(data, test_years, variables, mean_dict, std_dict, lead_time_steps)

# 3. Predict
print("Loading model...")
filepath = 'IMDAA_MultiVar_Transformer_T850_5days.keras'
model = keras.models.load_model(filepath, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})

print("Predicting...")
pred_norm = model.predict(X_test, batch_size=32).squeeze()
pred_result = pred_norm * std_dict['TMP_prl'].values + mean_dict['TMP_prl'].values

target_ds = target_ds.isel(time=slice(lead_time_steps, None))

# 4. Plot
# Select a random interesting index
idx = 100 
actual_field = target_ds.isel(time=idx).values
pred_field = pred_result[idx]
error_field = pred_field - actual_field

time_label = str(target_ds.isel(time=idx).time.values)[:10]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Common arguments for temperature
vmin = min(actual_field.min(), pred_field.min())
vmax = max(actual_field.max(), pred_field.max())

im0 = axes[0].imshow(actual_field, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
axes[0].set_title(f'Ground Truth (T850) - {time_label}', fontsize=12, fontweight='bold')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(pred_field, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
axes[1].set_title(f'Multi-Var ViT Prediction', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(error_field, cmap='bwr', origin='lower', vmin=-np.max(np.abs(error_field)), vmax=np.max(np.abs(error_field)))
axes[2].set_title(f'Spatial Error (Prediction - Actual)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('geographical_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved geographical_heatmap.png")
