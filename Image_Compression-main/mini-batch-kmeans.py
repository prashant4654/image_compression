# from sklearn.cluster import MiniBatchKMeans
# from skimage.io import imread
# import matplotlib.pyplot as plt
# import numpy as np
# import imageio

# # Load image
# img = imread('college.png')
# original_shape = img.shape
# pixels = img.reshape(-1, 3)

# # Mini-Batch K-Means
# K = 16
# kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1000)
# kmeans.fit(pixels)
# labels = kmeans.predict(pixels)
# compressed_pixels = kmeans.cluster_centers_[labels]
# compressed_img = compressed_pixels.reshape(original_shape).astype(np.uint8)

# # Save compressed image
# imageio.imwrite('compressed_image.jpg', compressed_img)

# # Display
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title("Original")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(compressed_img)
# plt.title(f"Compressed with {K} colors")
# plt.axis('off')
# plt.show()



from sklearn.cluster import MiniBatchKMeans
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Load and flatten image
img = imread('college.png')  # Replace with your image path
original_shape = img.shape
pixels = img.reshape(-1, 3)

# Set MSE threshold
target_mse = 1  # Try adjusting this
found = False

# Try increasing K in steps
for K in range(4, 129, 4):  # Try K = 4, 8, ..., 64
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1000, random_state=42)
    kmeans.fit(pixels)
    
    labels = kmeans.predict(pixels)
    compressed_pixels = kmeans.cluster_centers_[labels]
    compressed_img = compressed_pixels.reshape(original_shape)

    # Calculate mean squared error
    mse = np.mean((img.astype(np.float64) - compressed_img) ** 2)
    print(f"K = {K}, MSE = {mse:.2f}")

    if mse < target_mse:
        print(f"✅ Acceptable compression found at K = {K} with MSE = {mse:.2f}")
        found = True
        break

if not found:
    print("❌ No acceptable compression found. Consider increasing the K range or loosening the error threshold.")

# Save and display the result
compressed_img = compressed_img.astype(np.uint8)
imageio.imwrite(f'auto_compressed_minibatch_K{K}.jpg', compressed_img)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title(f"Mini-Batch K-Means (K={K}, MSE={mse:.2f})")
plt.axis('off')

plt.show()
