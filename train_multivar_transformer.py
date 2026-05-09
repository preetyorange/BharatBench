import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import json
warnings.filterwarnings('ignore')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Data loading
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
    
    X_train_list, X_valid_list, X_test_list = [], [], []
    
    for var in vars_list:
        mean = train_data[var].mean()
        std = train_data[var].std()
        
        means[var] = mean
        stds[var] = std
        
        train_norm = (train_data[var] - mean) / std
        valid_norm = (valid_data[var] - mean) / std
        test_norm = (test_data[var] - mean) / std
        
        X_train_list.append(train_norm.isel(time=slice(None, -lead_steps)).values)
        X_valid_list.append(valid_norm.isel(time=slice(None, -lead_steps)).values)
        X_test_list.append(test_norm.isel(time=slice(None, -lead_steps)).values)
        
        if var == target_var:
            Y_train = train_norm.isel(time=slice(lead_steps, None)).values[..., None]
            Y_valid = valid_norm.isel(time=slice(lead_steps, None)).values[..., None]
            Y_test = test_norm.isel(time=slice(lead_steps, None)).values[..., None]

    X_train = np.stack(X_train_list, axis=-1)
    X_valid = np.stack(X_valid_list, axis=-1)
    X_test = np.stack(X_test_list, axis=-1)
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, means, stds

print("Processing data...")
X_train, Y_train, X_valid, Y_valid, X_test, Y_test, means, stds = get_train_valid_test_dataset(lead_time_steps, data, variables, target_var)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

# Transformer Architecture
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

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Model parameters
patch_size = 4
num_patches = (32 // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 4
mlp_head_units = [2048, 1024]
num_channels = len(variables)

inputs = layers.Input(shape=(32, 32, num_channels))
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

representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
x = layers.Reshape((32 // patch_size, 32 // patch_size, projection_dim))(representation)
x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="swish")(x)
x = layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="swish")(x)
outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

print("Building Multi-Variable Transformer...")
model.summary()

filepath = 'IMDAA_MultiVar_Transformer_T850_5days.keras'
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)

print("Training model...")
history = model.fit(
    X_train, Y_train,
    epochs=15,
    validation_data=(X_valid, Y_valid),
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop, checkpoint]
)
