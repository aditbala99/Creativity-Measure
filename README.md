# Dino V2 and KMeans Integration

This repository presents an integration of the Dino V2 model for image embeddings with KMeans clustering for novelty detection in image data. The process involves data preparation, Dino V2 embeddings, and KMeans clustering with additional novelty and surprise metrics.

## Dino V2 Code Explanation

1. **Data Preparation**: Begin by uploading your image data to your Google Drive and establish a link to your Google Colab notebook.

2. **Image Extraction**: Extract the image files from the specified folder into the Colab working directory.

3. **Data List Creation**: The 'data' list will store the file paths of all PNG files found in the specified directory and its subdirectories.

4. **CUDA Configuration**: Create a symbolic link (soft link) to change the default CUDA version to CUDA 10, ensuring compatibility with the Dino V2 model.

5. **GPU Setup**: Set up CUDA on a GPU to leverage hardware acceleration.

6. **Requirements**: Add the 'requirements.txt' file to your Colab directory and download the necessary packages to use DINO embeddings. You can find the requirements file [here](https://raw.githubusercontent.com/facebookresearch/dinov2/main/requirements.txt).

7. **Model Selection**: Choose the Dino V2 model to be used. In this project, 'dinov2_vitg14,' referring to the Vision Transformer (ViT) version of the Dino V2 model, is employed.

8. **Data Preprocessing**: Preprocess the dataset, including resizing images, converting to tensors, and normalizing pixel values.

9. **Forward Pass**: Perform a forward pass through the pre-trained Dino V2 model for each image in the dataset, storing the output embeddings in an 'embeddings' list along with image file paths.

10. **Conversion to NumPy**: Convert the PyTorch tensor embeddings to NumPy arrays.

11. **Embedding Storage**: Save the embeddings to a folder in your Colab notebook for future use in novelty prediction and clustering.

## KMeans Code Explanation

### Building the KMeans Model

1. **Data Upload**: Upload your Dino embeddings data to Google Drive and link it to your Google Colab notebook.

2. **Embedding Extraction**: Extract the embedding files from the specified folder in the Colab working directory.

3. **Data List Creation**: The 'data' list contains the file paths of '.npy' files in the specified directory and subdirectories.

4. **Data Splitting**: Split the data into training and test datasets, using an 80/20 ratio. Apply the KMeans model to the training set.

5. **PCA Visualization**: Visualize clustering results using Principal Component Analysis (PCA) based on the training set.

6. **Optimal Cluster Count**: Determine the optimal number of clusters (k) using the elbow method.

7. **KMeans Re-Run**: Re-run the KMeans model on the training set with the optimal k value.

8. **Label Generation**: Apply the trained KMeans model to generate labels for the test dataset.

9. **Visualization**: Plot a bar graph to display the distribution of clusters based on the test labels.

### Building the Novelty Metrics Model

1. **Average Distance Calculation**: Calculate the average distance of samples to their respective cluster centroids.

2. **Novelty Detection**: Identify novel samples based on distances from their cluster centroids.

### Building the Surprise Metrics Model

1. **Pairwise Distance Calculation**: Calculate pairwise distances between centroids in the KMeans model.

2. **Average Centroid Distance**: Calculate the average of pairwise distances between centroids.

3. **Test Sample Assignment**: Assign new test samples to the nearest cluster centroids.

4. **Surprise Detection**: Identify surprising samples based on distances from cluster centroids.

This integration enables effective image embedding, clustering, and novelty detection using the Dino V2 model combined with KMeans clustering and surprise detection metrics.
