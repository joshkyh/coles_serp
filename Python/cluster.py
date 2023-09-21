import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin_min
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def kmeans_cosine(X, n_clusters, max_iter=100, tol=1e-4):
    X_norm = normalize(X, axis=1, norm='l2')
    rng = np.random.RandomState(42)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    prev_avg_cosine_distance = np.inf
    for i in range(max_iter):
        labels, distances = pairwise_distances_argmin_min(X_norm, centers, metric='cosine')
        new_centers = np.array([X[labels == j].mean(0) for j in range(n_clusters)])
        avg_cosine_distance = np.mean(distances)
        if np.abs(prev_avg_cosine_distance - avg_cosine_distance) < tol:
            break
        prev_avg_cosine_distance = avg_cosine_distance
        centers = new_centers
    return labels, centers, avg_cosine_distance

def tune_cluster() -> pd.DataFrame:

    # Read the reviews DataFrame from a Parquet file
    reviews = pd.read_parquet('data/google_maps_reviews_with_embeddings.parquet', engine='pyarrow')


    # Read the google maps Datafram
    google_maps = pd.read_parquet('data/google_maps_results.parquet', engine='pyarrow')

    # Join google_maps into reviews
    reviews = reviews.merge(google_maps, on='data_id')

    # Convert the 'embedding' column to a NumPy array
    embedding_array = np.stack(reviews['embedding'].to_numpy())

    # Modify the kmeans_cosine function to return average cosine distance


    # Run k-means for different numbers of clusters and store average cosine distance
    n_clusters_range = range(2, 31)
    avg_cosine_distances = []

    for n_clusters in n_clusters_range:
        _, _, avg_cosine_distance = kmeans_cosine(embedding_array, n_clusters=n_clusters)
        avg_cosine_distances.append(avg_cosine_distance)



    # Create the plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(n_clusters_range), y=avg_cosine_distances, mode='lines+markers'))

    fig.update_layout(
        title='Elbow Method for Optimal Number of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='Average Cosine Distance',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )


    # Save the plot as a PNG file
    fig.write_image('visualization/elbow_method.png')

def get_cluster() -> pd.DataFrame:

    # Read the reviews DataFrame from a Parquet file
    reviews = pd.read_parquet('data/google_maps_reviews_with_embeddings.parquet', engine='pyarrow')


    # Read the google maps Datafram
    google_maps = pd.read_parquet('data/google_maps_results.parquet', engine='pyarrow')

    # Join google_maps into reviews
    reviews = reviews.merge(google_maps, on='data_id')

    # Convert the 'embedding' column to a NumPy array
    embedding_array = np.stack(reviews['embedding'].to_numpy())


    # Run k-means clustering on original high-dimensional embeddings
    labels, centers, avg_cosine_distance = kmeans_cosine(embedding_array, n_clusters=15)

    # Add the cluster labels to the DataFrame
    reviews['cluster_label'] = labels

    return reviews

if __name__ == '__main__':
    reviews = get_cluster()

    # Group by 'cluster_label' and apply a lambda function to concatenate the top 10 'snippet' strings
    top_X_snippets_per_cluster = reviews.groupby('cluster_label')['snippet'].apply(
        lambda x: ' || '.join(x.head(12))
    ).reset_index()

    # Rename the columns for clarity
    top_X_snippets_per_cluster.columns = ['cluster_label', 'concatenated_snippets']

    print(top_X_snippets_per_cluster)

    top_X_snippets_per_cluster['theme'] = ''
    top_X_snippets_per_cluster['theme'].at[0] = 'Mixed reviews: spacious but disorienting layout, good location'
    top_X_snippets_per_cluster['theme'].at[1] = 'Price and variety-focused, inconsistent product availability'
    top_X_snippets_per_cluster['theme'].at[2] = 'Poor customer service and payment issues for online orders'
    top_X_snippets_per_cluster['theme'].at[3] = 'Great shopping experience in diverse settings'
    top_X_snippets_per_cluster['theme'].at[4] = 'Large, extensive supermarket with global appeal'
    top_X_snippets_per_cluster['theme'].at[5] = 'Busy but convenient, focuses on essentials'
    top_X_snippets_per_cluster['theme'].at[6] = 'In need of renovation, issues with cleanliness and lighting'
    top_X_snippets_per_cluster['theme'].at[7] = 'Poor inventory, out-of-date products, inadequate labeling'
    top_X_snippets_per_cluster['theme'].at[8] = 'Late-night operation, Korean reviews highlight value'
    top_X_snippets_per_cluster['theme'].at[9] = 'General positivity with room for improvement'
    top_X_snippets_per_cluster['theme'].at[10] = 'Negative work environment, customer service complaints'
    top_X_snippets_per_cluster['theme'].at[11] = 'Expensive and poorly organized, varying stock levels'
    top_X_snippets_per_cluster['theme'].at[12] = 'Well-stocked with excellent customer service'
    top_X_snippets_per_cluster['theme'].at[13] = 'Multiple issues with food quality and staff behavior'
    top_X_snippets_per_cluster['theme'].at[14] = 'Excellent customer service, staff highly praised'


    # Join the 'theme' column to the reviews DataFrame
    reviews = reviews.merge(top_X_snippets_per_cluster[['cluster_label', 'theme']], on='cluster_label')

    # Plots
    embedding_array = np.stack(reviews['embedding'].to_numpy())

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2)
    reviews['tsne'] = list(tsne.fit_transform(embedding_array))

    # Convert the t-SNE results to DataFrame columns for easier plotting
    tsne_results = np.array(reviews['tsne'].tolist())
    reviews['tsne_1'] = tsne_results[:, 0]
    reviews['tsne_2'] = tsne_results[:, 1]

    # rename rating_x to review_rating
    reviews = reviews.rename(columns={'rating_x': 'review_rating'})

    fig = px.scatter(reviews,
                     x='tsne_1',
                     y='tsne_2',
                     color='cluster_label',
                     title='Interactive Scatter Plot of t-SNE Clusters',
                     labels={'tsne_1': 't-SNE 1', 'tsne_2': 't-SNE 2'},
                     hover_data=['address', 'review_rating', 'snippet', 'theme', 'date'],
                     template='plotly_white')

    # Export
    fig.write_html('visualization/2d_tsne_clusters.html')



