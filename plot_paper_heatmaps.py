import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import ssl
import geopandas as gpd

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# --- CUSTOM LAYERS FOR VANILLA AND MULTIVAR VIT ---
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

# --- CUSTOM LAYERS FOR SWIN VIT ---
def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, window_size, window_size, C])

def window_reverse(windows, window_size, H, W, C):
    B = tf.shape(windows)[0] // ( (H * W) // (window_size * window_size) )
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [B, H, W, C])

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=True)
        self.proj = layers.Dense(dim)
        
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_relative_distance, num_heads),
            initializer="zeros", trainable=True, name="relative_position_bias_table")
            
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
            trainable=False, dtype=tf.int32, name="relative_position_index")

    def call(self, x, mask=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1]))
        relative_position_bias = tf.reshape(relative_position_bias, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)
        if mask is not None:
            nW = tf.shape(mask)[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + mask_float
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, C])
        return self.proj(x)
        
    def get_config(self):
        config = super(WindowAttention, self).get_config()
        config.update({'dim': self.dim, 'window_size': self.window_size, 'num_heads': self.num_heads})
        return config

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
        self.mlp = keras.Sequential([layers.Dense(dim * 4, activation=tf.nn.gelu), layers.Dense(dim)])

    def call(self, x):
        H, W = tf.shape(x)[1], tf.shape(x)[2]
        B, C = tf.shape(x)[0], tf.shape(x)[3]
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
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
        
    def get_config(self):
        config = super(SwinTransformerBlock, self).get_config()
        config.update({'dim': self.dim, 'num_heads': self.num_heads, 'window_size': self.window_size, 'shift_size': self.shift_size})
        return config

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, x):
        B = tf.shape(x)[0]
        return tf.image.extract_patches(
            images=x, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
    def get_config(self):
        config = super(PatchExtract, self).get_config()
        config.update({'patch_size': self.patch_size})
        return config


# --- DATA LOADING ---
print("Loading dataset...")
data = xr.open_dataset('dataset-bharatbench/IMDAA_merged_1.08_1990_2020.nc')

train_years = slice('1990', '2017')
test_years = slice('2019', '2020')
lead_time_steps = 20
variables = ['HGT_prl', 'TMP_prl', 'TMP_2m', 'APCP_sfc']

mean_dict, std_dict = {}, {}
for var in variables:
    train_data = data[var].sel(time=train_years)
    mean_dict[var] = train_data.mean()
    std_dict[var] = train_data.std()

def get_multi_inputs(subset_data):
    channels = []
    for var in variables:
        norm_var = (subset_data[var] - mean_dict[var]) / std_dict[var]
        channels.append(norm_var.isel(time=slice(None, -lead_time_steps)).values)
    return np.stack(channels, axis=-1)

subset = data.sel(time=test_years)

# Vanilla & Swin inputs (Only TMP_prl)
X_single = ((subset['TMP_prl'] - mean_dict['TMP_prl']) / std_dict['TMP_prl']).isel(time=slice(None, -lead_time_steps)).values[..., None]
# MultiVar inputs
X_multi = get_multi_inputs(subset)
# Targets
target_ds = data['TMP_prl'].sel(time=test_years).isel(time=slice(lead_time_steps, None))

idx = 100 
X_single_sample = X_single[idx:idx+1]
X_multi_sample = X_multi[idx:idx+1]

# --- INFERENCE ---
print("Running Inference...")
custom_vit = {'Patches': Patches, 'PatchEncoder': PatchEncoder}
custom_swin = {'WindowAttention': WindowAttention, 'SwinTransformerBlock': SwinTransformerBlock, 'PatchExtract': PatchExtract}

model_vanilla = keras.models.load_model('IMDAA_Transformer_T850_5days.keras', custom_objects=custom_vit)
model_swin = keras.models.load_model('IMDAA_Swin_Transformer_T850_5days.keras', custom_objects=custom_swin)
model_multi = keras.models.load_model('IMDAA_MultiVar_Transformer_T850_5days.keras', custom_objects=custom_vit)

pred_vanilla = model_vanilla.predict(X_single_sample, verbose=0).squeeze() * std_dict['TMP_prl'].values + mean_dict['TMP_prl'].values
pred_swin = model_swin.predict(X_single_sample, verbose=0).squeeze() * std_dict['TMP_prl'].values + mean_dict['TMP_prl'].values
pred_multi = model_multi.predict(X_multi_sample, verbose=0).squeeze() * std_dict['TMP_prl'].values + mean_dict['TMP_prl'].values

actual_field = target_ds.isel(time=idx).values
time_label = str(target_ds.isel(time=idx).time.values)[:10]
lats = target_ds.latitude.values
lons = target_ds.longitude.values

# --- PLOTTING ---
print("Generating Plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

titles = ['(a) Ground Truth (T850)', '(b) Vanilla ViT Prediction', '(c) Swin Transformer Prediction', '(d) Multi-Variable ViT Prediction']
fields = [actual_field, pred_vanilla, pred_swin, pred_multi]

vmin = min([f.min() for f in fields])
vmax = max([f.max() for f in fields])

# HARDCODED GEOGRAPHIC EXTENT TO FIX GRADIENT
extent = [65, 100, 5, 40]

try:
    india = gpd.read_file('india_boundary.geojson')
except Exception as e:
    print(f"Failed to load boundary: {e}")
    india = None

for i, ax in enumerate(axes):
    ax.set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    
    # Official Indian Border
    if india is not None:
        india.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, transform=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Using imshow instead of pcolormesh to ensure gradient renders properly
    im = ax.imshow(fields[i], transform=ccrs.PlateCarree(), cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin='lower')
    
    ax.set_title(titles[i], loc='left', fontsize=14, fontweight='bold', pad=10)

# Add a single colorbar for the whole figure
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Temperature (K)', fontsize=12, fontweight='bold')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.suptitle(f'T850 Geographical Evaluation ({time_label})', fontsize=20, fontweight='bold', y=0.95)

plt.savefig('paper_style_heatmaps.png', dpi=300, bbox_inches='tight')
print("Successfully generated paper_style_heatmaps.png")
