# from sklearn.cluster import KMeans
# from skimage.io import imread
# import matplotlib.pyplot as plt
# import numpy as np
# import imageio

# # Load image
# img = imread('college.png')  # Replace with your image path
# original_shape = img.shape

# # Reshape image to (num_pixels, 3)
# pixels = img.reshape(-1, 3)

# # Apply K-Means clustering
# K = 16  # Number of colors
# kmeans = KMeans(n_clusters=K, random_state=42)
# kmeans.fit(pixels)

# # Assign each pixel to the nearest cluster center
# labels = kmeans.predict(pixels)
# compressed_pixels = kmeans.cluster_centers_[labels]
# compressed_img = compressed_pixels.reshape(original_shape).astype(np.uint8)

# # Save compressed image
# imageio.imwrite('compressed_kmeans.jpg', compressed_img)

# # Display original and compressed images
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(compressed_img)
# plt.title(f"K-Means Compressed ({K} colors)")
# plt.axis('off')

# plt.show()



from sklearn.cluster import KMeans
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Load and flatten image
img = imread('college.png')  # Replace with actual path
original_shape = img.shape
pixels = img.reshape(-1, 3)

# Set your target MSE
target_mse = 1  # Try changing this
found = False

# Loop over K until desired error is achieved
for K in range(4, 129, 4):  # Try K = 4, 8, ..., 128
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.predict(pixels)
    compressed_pixels = kmeans.cluster_centers_[labels]
    compressed_img = compressed_pixels.reshape(original_shape)

    # Compute MSE (Mean Squared Error)
    mse = np.mean((img.astype(np.float64) - compressed_img) ** 2)
    print(f"K = {K}, MSE = {mse:.2f}")

    if mse < target_mse:
        print(f"✅ Acceptable compression achieved at K = {K} with MSE = {mse:.2f}")
        found = True
        break

if not found:
    print("❌ No acceptable compression found. Consider increasing the K range or loosening the error threshold.")

# Save and show the final image
compressed_img = compressed_img.astype(np.uint8)
imageio.imwrite(f'auto_compressed_kmeans_K{K}.jpg', compressed_img)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title(f"K-Means (K={K}, MSE={mse:.2f})")
plt.axis('off')

plt.show()
