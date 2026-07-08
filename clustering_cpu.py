import os
import torch
import numpy as np
import faiss
import argparse
import time


def train_kmeans_faiss_cpu(
    data,
    save_path,
    n_clusters,
    n_iter=20,
    verbose=True,
    min_points=32,
    max_points=1024
):
    """
    Runs K-means using FAISS on CPU.

    Args:
        data (np.ndarray): Data array to cluster, shape (num_samples, num_features).
        save_path (str): Directory where centroids.npy will be saved.
        n_clusters (int): Number of clusters to form.
        n_iter (int): Number of K-means iterations.
        verbose (bool): Whether to print FAISS clustering output.
        min_points (int): Minimum points per centroid.
        max_points (int): Maximum points per centroid.

    Returns:
        faiss.IndexFlatL2: CPU FAISS index used for training.
    """
    import os
    import numpy as np
    import faiss

    # Ensure data is float32 and contiguous for FAISS
    data = np.ascontiguousarray(data.astype(np.float32))

    print("Using CPU for K-means clustering")

    dim = data.shape[1]

    # Set up FAISS K-means clustering
    kmeans = faiss.Clustering(dim, n_clusters)
    kmeans.niter = n_iter
    kmeans.verbose = verbose
    kmeans.max_points_per_centroid = max_points
    kmeans.min_points_per_centroid = min_points

    # CPU index
    cpu_index = faiss.IndexFlatL2(dim)

    # Train K-means on CPU
    kmeans.train(data, cpu_index)

    # Extract and save centroids
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(n_clusters, dim)

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "centroids.npy"), centroids)

    print(f"Centroids shape: {centroids.shape}")

    return cpu_index


def treat_one_sample(data_dir, save_path, n_clusters, n_iter=20, min_points=32, max_points=1024):
    """
    Main function to load data, concatenate, and perform K-means clustering.
    
    Args:
        data_dir (str): Directory containing the .pt, .pth, or .npy files.
        n_clusters (int): Number of clusters for K-means.
        n_iter (int): Number of iterations for K-means.
    """
    tdeb_sample = time.time()
    os.makedirs(save_path, exist_ok=True)
    # Get all file paths in the directory
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith(('.pt', '.pth', '.npy'))]
    # Number of points to load per file
    number_to_load = (max_points*n_clusters)
    print(f"Number of points to load: {number_to_load}")
    tr=0
    data = []
    n_points = 0
    while n_points<number_to_load and tr<len(file_paths):
        if file_paths[tr].endswith(('.pt', '.pth')):
            # Load PyTorch tensor and convert to numpy array
            dat = torch.load(file_paths[tr],map_location=torch.device("cpu")).cpu().numpy()
            
            if number_to_load-n_points>len(dat):
                data.append(dat)
            else:
                data.append(dat[:number_to_load-n_points])
        elif file_paths[tr].endswith('.npy'):
            # Load numpy array directly
            dat = np.load(file_paths[tr])[:number_to_load]
            if number_to_load-n_points>len(dat):
                data.append(dat)
            else:
                data.append(dat[:number_to_load-n_points])
        else:
            raise ValueError(f"Unsupported file format: {file_paths[tr]}")
        n_points+=len(dat)
        tr+=1
    data = np.concatenate(data, axis=0)
    print(f"Data shape after concatenation: {data.shape}")
    print(f"Loading time: {time.time()-tdeb_sample}")
    index = train_kmeans_faiss_cpu(data, save_path, n_clusters, n_iter, True, min_points, max_points)
    
        

def main(data_dir, save_path,n_clusters, n_iter=20, min_points=32, max_points=1024):
    save_path = os.path.join(save_path, str(n_clusters))
    os.makedirs(save_path, exist_ok=True)
    tdeb = time.time()
    c=0
    for sample in os.listdir(data_dir):
        ttemp = time.time()
        ## Option to skip already treated samples in case of a crash or interruption
        if sample in os.listdir(save_path):
            print(sample,"already treated")
            c+=1
            continue
        print(c)
        treat_one_sample(os.path.join(data_dir,sample), os.path.join(save_path,sample), n_clusters, n_iter, min_points, max_points)
        print(f"Sample time: {time.time()-ttemp}")
    print(f"Total time: {time.time()-tdeb}")
    print(f"Time by sample: {(time.time()-tdeb)/len(os.listdir(data_dir))}")    

if __name__ == "__main__":
    parsearg = argparse.ArgumentParser()
    parsearg.add_argument('data_dir', type=str, help='Directory containing the data files.')
    parsearg.add_argument('save_path', type=str, help='Directory to save the clustering results.')
    parsearg.add_argument('n_clusters', type=int, help='Number of clusters for K-means.')
    parsearg.add_argument('n_iter', type=int, default=20, help='Number of iterations for K-means.')
    parsearg.add_argument('min_points', type=int, default=32, help='Minimum number of points per centroid.')
    parsearg.add_argument('max_points', type=int, default=1024, help='Maximum number of points per centroid.')
    args = parsearg.parse_args()
    main(args.data_dir, args.save_path, args.n_clusters, args.n_iter, args.min_points, args.max_points)
