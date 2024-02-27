Summary file for clustering analysis

> Necesary data for the analysis
>> Need three_pop_mpi/data
>> Run postprocess.py before

# Prepare dataset for clustering
> Use three_pop_mpi/data 
[Source code](./prepare_clustering_data.ipynb)

Rearrange post processed dataset and normalize dataset using Z-score normalization method. Then, split dataset into _sub_ and _total_ population measured 

# Test clustering (testing various clustering method for selecting the clustering method)
> Use "./data/align_data_sub.pkl" or "./data/align_data_tot.pkl" 
[Source code](./test_clustering.ipynb)

# Cluster the feature dataset
[source code](./cluster_oscillation_features.ipynb)
Check the point index interactively

## Need to run [explore_kmeans2.py](./explore_kmeans2.py)
> Get ./data/kmeans_pred.pkl



<!-- # Rearrange post processed dataset
[Source code](./align_postprocess_data.ipynb)
Flatten the dataset and check 

> **_output_**: ./data/align_data.pkl

# Select cluster features
[Source code](./clustering_feature_selection.ipynb) -->


check the covariance and variance of each dataset and export splitted data after Z-score normalization

# Data descrption (./data)

- kmeans_pred.pkl
    Kmeans clustering result from K = 3 to 30 with 100 iteraction for each
- kmeans_pred_inv.pkl
    Kmeans clustering result from K = 3 to 30 with 100 iteraction for each, tau -> 1/tau
- kmeans_pred_partial.pkl
    Kmeans clustering result from K = 3 to 30 with 100 iteraction for each, lead-lag ratio, dphi are removed
