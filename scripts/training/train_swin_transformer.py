import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import warnings
warnings.filterwarnings('ignore')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Data loading
print("Loading dataset...")
data = xr.open_dataset('dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc')
ds = data['TMP_prl'].to_dataset()

train_years = slice('1990', '2017')
valid_years = slice('2018', '2018')
test_years = slice('2019', '2020')
lead_time_steps = 20

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

print("Processing data...")
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
    acc = np.sum(pred_norm * act_norm) / np.sqrt(np.sum(pred_norm ** 2) * np.sum(act_norm ** 2))
    return acc

# Swin Transformer Components
def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W, C):
    B = tf.shape(windows)[0] // ( (H * W) // (window_size * window_size) )
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=True)
        self.proj = layers.Dense(dim)
        
        # relative position bias
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_relative_distance, num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index, dtype=tf.int32),
            trainable=False,
            dtype=tf.int32,
            name="relative_position_index"
        )

    def call(self, x, mask=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)

        relative_position_bias = tf.gather(
            self.relative_position_bias_table, 
            tf.reshape(self.relative_position_index, [-1])
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1],
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = tf.shape(mask)[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + mask_float
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, C])
        x = self.proj(x)
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size, shift_size, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([
            layers.Dense(dim * 4, activation=tf.nn.gelu),
            layers.Dense(dim),
        ])

    def call(self, x):
        H, W = tf.shape(x)[1], tf.shape(x)[2]
        B, C = tf.shape(x)[0], tf.shape(x)[3]
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
            
            # calculate attention mask for shifted window
            img_mask = np.zeros((1, x.shape[1], x.shape[2], 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, 0.0)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, C])

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        B = tf.shape(x)[0]
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return patches

def create_swin_model():
    inputs = layers.Input(shape=(32, 32, 1))
    
    # 32x32 -> 16x16
    x = PatchExtract(patch_size=2)(inputs)
    x = layers.Dense(64)(x)
    
    x = SwinTransformerBlock(dim=64, num_heads=4, window_size=4, shift_size=0)(x)
    x = SwinTransformerBlock(dim=64, num_heads=4, window_size=4, shift_size=2)(x)
    x = SwinTransformerBlock(dim=64, num_heads=4, window_size=4, shift_size=0)(x)
    x = SwinTransformerBlock(dim=64, num_heads=4, window_size=4, shift_size=2)(x)
    
    # 16x16 -> 32x32
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="swish")(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="linear")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

print("Building Swin Transformer...")
model = create_swin_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5), loss='mse')
model.summary()

filepath = 'IMDAA_Swin_Transformer_T850_5days.keras'
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

print("Evaluating...")
# Use the trained model directly since custom layers might be tricky to load simply without registering all of them
# model is already the best model since we could load weights, or just use the model object
model.load_weights(filepath)

target = ds.sel(time=test_years)
pred_test = X_test[:, :, :, 0].copy()
pred_test[:] = model.predict(X_test, batch_size=32).squeeze()

pred_result = pred_test * std.TMP_prl.values + mean.TMP_prl.values
pred_result = xr.DataArray(
    pred_result, 
    dims=target.isel(time=slice(lead_time_steps, None)).dims, 
    coords=target.isel(time=slice(lead_time_steps, None)).coords
)

rmse = compute_rmse(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values
mae = compute_mae(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values
acc = compute_acc(pred_result, target.isel(time=slice(lead_time_steps, None))).TMP_prl.values

results = {
    'RMSE': float(rmse),
    'MAE': float(mae),
    'ACC': float(acc)
}
with open('swin_metrics.json', 'w') as f:
    json.dump(results, f)

print(f"Final Swin Metrics: RMSE={float(rmse):.4f}, MAE={float(mae):.4f}, ACC={float(acc):.4f}")
