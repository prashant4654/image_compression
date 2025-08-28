from sklearn.mixture import GaussianMixture
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Load and flatten image
img = imread('college.png')  # Replace with your image path
original_shape = img.shape
pixels = img.reshape(-1, 3)

# Set your target MSE threshold
target_mse = 300  # Adjust as needed
found = False

# Try increasing K in steps
for K in range(4, 129, 4):  # Try K = 4, 8, ..., 64
    gmm = GaussianMixture(n_components=K, covariance_type='tied', random_state=42)
    gmm.fit(pixels)

    labels = gmm.predict(pixels)
    compressed_pixels = gmm.means_[labels]
    compressed_img = compressed_pixels.reshape(original_shape)

    # Compute MSE
    mse = np.mean((img.astype(np.float64) - compressed_img) ** 2)
    print(f"K = {K}, MSE = {mse:.2f}")

    if mse < target_mse:
        print(f"✅ Acceptable compression achieved at K = {K} with MSE = {mse:.2f}")
        found = True
        break

if not found:
    print("❌ No acceptable compression found. Try increasing max K or loosening error threshold.")

# Save and display the final compressed image
compressed_img = compressed_img.astype(np.uint8)
imageio.imwrite(f'auto_compressed_gmm_K{K}.jpg', compressed_img)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title(f"GMM Compressed (K={K}, MSE={mse:.2f})")
plt.axis('off')

plt.show()
