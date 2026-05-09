import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import json
warnings.filterwarnings('ignore')

print("Loading dataset...")
data = xr.open_dataset('dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc')

variables = ['HGT_prl', 'TMP_prl', 'TMP_2m', 'APCP_sfc']
target_var = 'TMP_prl'
target_idx = variables.index(target_var)

train_years = slice('1990', '2017')
valid_years = slice('2018', '2018')
test_years = slice('2019', '2020')
lead_time_steps = 20

def get_train_valid_test_dataset(lead_steps, data, vars_list, target_var):
    train_data = data.sel(time=train_years)
    valid_data = data.sel(time=valid_years)
    test_data = data.sel(time=test_years)
    
    means = {}
    stds = {}
    
    X_test_list = []
    
    for var in vars_list:
        mean = train_data[var].mean()
        std = train_data[var].std()
        
        means[var] = mean
        stds[var] = std
        
        test_norm = (test_data[var] - mean) / std
        X_test_list.append(test_norm.isel(time=slice(None, -lead_steps)).values)

    X_test = np.stack(X_test_list, axis=-1)
    return X_test, means, stds

X_test, means, stds = get_train_valid_test_dataset(lead_time_steps, data, variables, target_var)

def compute_rmse(prediction, actual,  mean_dims = ('time', 'latitude', 'longitude')):
    error = prediction - actual
    rmse = np.sqrt(((error)**2 ).mean(mean_dims))
    return rmse

def compute_mae(prediction, actual, mean_dims = ('time', 'latitude', 'longitude')):
    error = prediction - actual
    mae = np.abs(error).mean(mean_dims)
    return mae

def compute_acc(prediction, actual):
    clim = actual.mean('time')
    try:
        t = np.intersect1d(prediction.time, actual.time)
        pred_anomaly = prediction.sel(time=t) - clim
    except AttributeError:
        t = actual.time.values
        pred_anomaly = prediction - clim
    act_anomaly = actual.sel(time=t) - clim
    
    pred_norm = pred_anomaly - pred_anomaly.mean()
    act_norm = act_anomaly - act_anomaly.mean()

    acc = (
            np.sum(pred_norm * act_norm) /
            np.sqrt(
                np.sum(pred_norm ** 2) * np.sum(act_norm ** 2)
            )
    )
    return acc

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
        
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
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return config

print("Evaluating...")
filepath = 'IMDAA_MultiVar_Transformer_T850_5days.keras'
best_model = keras.models.load_model(filepath, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})

target = data[target_var].sel(time=test_years)
pred_test = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
pred_test[:] = best_model.predict(X_test, batch_size=32).squeeze()[..., None]

pred_result = pred_test.squeeze() * stds[target_var].values + means[target_var].values
pred_result = xr.DataArray(
    pred_result, 
    dims=target.isel(time=slice(lead_time_steps, None)).dims, 
    coords=target.isel(time=slice(lead_time_steps, None)).coords
)

rmse_val = compute_rmse(pred_result, target.isel(time=slice(lead_time_steps, None))).values
mae_val = compute_mae(pred_result, target.isel(time=slice(lead_time_steps, None))).values
acc_val = compute_acc(pred_result, target.isel(time=slice(lead_time_steps, None))).values

rmse = float(rmse_val)
mae = float(mae_val)
acc = float(acc_val)

results = {
    'RMSE': rmse,
    'MAE': mae,
    'ACC': acc
}
with open('multivar_metrics.json', 'w') as f:
    json.dump(results, f)

print(f"Final Multi-Variable Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, ACC={acc:.4f}")
