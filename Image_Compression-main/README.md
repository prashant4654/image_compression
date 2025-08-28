# Image Compression Using Machine Learning

Image compression is compressing Image to lower quality from Original Quality. So, that less space should be ocupied in database.
So, By this We can say Many stuff can be deal through Machine learning Algorithms.

The web is loaded up with enormous measures of information as pictures. Individuals transfer a great many pictures each day via online media locales, for example, Instagram, Facebook and distributed storage stages, for example, google drive, and so on. With such a lot of information, image compression techniques become important to compress the images and reduce storage space. 

An image is made up of several intensity values known as Pixels. In a colored image, each pixel is of 3 bytes containing RGB (Red-Blue-Green) values having Red intensity value, then Blue and then Green intensity value for each pixel.

*****************************************************************************************************************************************************************

This project demonstrates how to compress images using various unsupervised clustering algorithms in Python. The goal is to reduce the number of unique colors in the image (i.e., compress its color space) while retaining as much visual quality as possible.

---

## 📌 Overview

We use the following clustering algorithms for image compression:

| Algorithm                    | Type            | Pros                           | Use Case              |
| ---------------------------- | --------------- | ------------------------------ | --------------------- |
| K-Means                      | Hard clustering | Fast, simple, widely supported | Standard compression  |
| Mini-Batch K-Means           | Hard clustering | Faster for large images        | Real-time compression |
| Gaussian Mixture Model (GMM) | Soft clustering | More flexible color modeling   | Complex scenes        |

---

## 🖼️ How It Works

1. Load the image using `skimage.io.imread`.
2. Reshape the image from `(height, width, 3)` to `(num_pixels, 3)`.
3. Cluster the pixel RGB values using the chosen algorithm.
4. Replace each pixel's color with the centroid (or weighted mean) of its cluster.
5. Reshape back to original image dimensions and save/display the compressed result.

---

## 📂 Project Structure

```
.
├── kmeans_compression.py
├── minibatch_compression.py
├── gmm_compression.py
├── sample_images/
│   └── your_image.jpg
├── auto_k_selection/
│   └── (optional looped K optimization)
└── compressed_outputs/
    ├── compressed_kmeans_K16.jpg
    ├── compressed_minibatch_K16.jpg
    └── compressed_gmm_K16.jpg
```

---

## 🚀 Algorithms Used

### 📌 1. K-Means Clustering

- Clusters similar pixel values using Euclidean distance.
- Fast and works well for general compression tasks.
- Can reduce an image to **K colors** effectively.

### 📌 2. Mini-Batch K-Means

- Same as K-Means but uses mini-batches for faster convergence.
- Useful for high-resolution or large datasets.
- Much quicker with similar visual results.

### 📌 3. Gaussian Mixture Models (GMM)

- Uses probabilistic assignments for clustering (soft clustering).
- More flexible than K-Means for overlapping color distributions.
- Slightly slower, but better for visually complex images.

---

## 🔄 Auto-K Optimization

All 3 implementations optionally include a loop that:

- Starts with a small K (e.g., 4)
- Increments K
- Stops when the **mean squared error (MSE)** drops below a user-defined threshold
- Saves the compressed image

---

## 📦 Requirements

- Python 3.6+
- scikit-learn
- scikit-image
- matplotlib
- imageio

```bash
pip install scikit-learn scikit-image matplotlib imageio
```

---

## 📸 Example Outputs

| Original Image | K-Means | Mini-Batch | GMM |
| -------------- | ------- | ---------- | --- |
|                |         |            |     |

---

:)

