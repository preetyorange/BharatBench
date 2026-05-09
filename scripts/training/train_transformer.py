import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import json
import argparse

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='TMP_prl', help='Target variable to predict')
args = parser.parse_args()
target_var = args.target

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load Data
print(f"Loading dataset for {target_var}...")
data = xr.open_dataset('dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc')
ds = data[target_var].to_dataset()

# Splits
train_years = slice('1990', '2017')
valid_years = slice('2018', '2018')
test_years = slice('2019', '2020')
lead_time_steps = 20

# Pipeline
print("Processing data...")
def get_train_valid_test_dataset(lead_steps, Data_array):
    train_data = Data_array.sel(time=train_years)
    valid_data = Data_array.sel(time=valid_years)
    test_data = Data_array.sel(time=test_years)

    mean = train_data.mean()
    std = train_data.std()

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    test_data = (test_data - mean) / std

    X_train = train_data[list(Data_array)[0]].isel(time=slice(None, -lead_steps)).values[..., None]
    Y_train = train_data[list(Data_array)[0]].isel(time=slice(lead_steps, None)).values[..., None]
    X_valid = valid_data[list(Data_array)[0]].isel(time=slice(None, -lead_steps)).values[..., None]
    Y_valid = valid_data[list(Data_array)[0]].isel(time=slice(lead_steps, None)).values[..., None]
    X_test = test_data[list(Data_array)[0]].isel(time=slice(None, -lead_steps)).values[..., None]
    Y_test = test_data[list(Data_array)[0]].isel(time=slice(lead_steps, None)).values[..., None]
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, mean, std

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, mean, std = get_train_valid_test_dataset(lead_time_steps, ds)

# Evaluation metrics
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
    return float(acc)

# Architecture
image_size = 32
patch_size = 4
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 4

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

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

def create_vit_model():
    inputs = layers.Input(shape=(image_size, image_size, 1))
    
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    grid_size = image_size // patch_size
    reshaped = layers.Reshape((grid_size, grid_size, projection_dim))(encoded_patches)
    
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="swish")(reshaped)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="swish")(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="linear")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

print("Building model...")
model = create_vit_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

print(f"Training model for {target_var}...")
filepath = f'IMDAA_Transformer_{target_var}_5days.keras'
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)

history = model.fit(
    X_train, Y_train,
    epochs=15,
    validation_data=(X_valid, Y_valid),
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop, checkpoint]
)

print("Evaluating...")
best_model = keras.models.load_model(filepath, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})

target = ds.sel(time=test_years)
pred_test = X_test[:, :, :, 0].copy()
pred_test[:] = best_model.predict(X_test, batch_size=32).squeeze()

pred_result = pred_test * getattr(std, target_var).values + getattr(mean, target_var).values
pred_result = xr.DataArray(
    pred_result, 
    dims=target.isel(time=slice(lead_time_steps, None)).dims, 
    coords=target.isel(time=slice(lead_time_steps, None)).coords
)

rmse = compute_rmse(pred_result, target.isel(time=slice(lead_time_steps, None)))
rmse = getattr(rmse, target_var).values

mae = compute_mae(pred_result, target.isel(time=slice(lead_time_steps, None)))
mae = getattr(mae, target_var).values

acc = compute_acc(pred_result, target.isel(time=slice(lead_time_steps, None)))
acc = getattr(acc, target_var).values

results = {
    'RMSE': float(rmse),
    'MAE': float(mae),
    'ACC': float(acc)
}
with open(f'transformer_metrics_{target_var}.json', 'w') as f:
    json.dump(results, f)

print(f"Final Metrics for {target_var}: RMSE={rmse:.4f}, MAE={mae:.4f}, ACC={acc:.4f}")
