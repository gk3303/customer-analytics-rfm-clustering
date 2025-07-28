"""
Advanced Customer Segmentation using RFM Analysis and K-Means Clustering
Implementation: RFM scoring methodology combined with K-Means clustering algorithm
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = r'Data\sales_data_sample.csv'


def load_and_preprocess_data(file_path):
    dataset = pd.read_csv(file_path, encoding='unicode_escape')
    required_columns = ['CUSTOMERNAME', 'ORDERNUMBER', 'ORDERDATE', 'SALES']
    processed_data = dataset[required_columns]
    processed_data['ORDERDATE'] = pd.to_datetime(processed_data['ORDERDATE'], errors='coerce')
    return processed_data


def build_rfm_metrics(dataframe, reference_point):
    rfm_metrics = dataframe.groupby('CUSTOMERNAME').agg({
        'ORDERDATE': lambda date: (reference_point - date.max()).days,
        'ORDERNUMBER': 'count',
        'SALES': 'sum'
    })
    rfm_metrics.rename(columns={
        'ORDERDATE': 'Recency', 
        'ORDERNUMBER': 'Frequency', 
        'SALES': 'MonetaryValue'
    }, inplace=True)
    return rfm_metrics


def compute_rfm_scores(dataframe):
    recency_score = pd.qcut(dataframe.Recency, 4, labels=list(range(0, 4)))
    frequency_score = pd.qcut(dataframe.Frequency, 4, labels=list(range(0, 4)))
    monetary_score = pd.qcut(dataframe.MonetaryValue, 4, labels=list(range(0, 4)))
    
    scored_rfm = pd.DataFrame({
        'Recency': recency_score, 
        'Frequency': frequency_score, 
        'MonetaryValue': monetary_score
    })
    return scored_rfm


def display_elbow_analysis(scored_rfm):
    rfm_values = scored_rfm.values
    cluster_analysis = []
    
    for cluster_count in range(1, 15):
        kmeans_model = KMeans(n_clusters=cluster_count)
        kmeans_model.fit(rfm_values)
        cluster_analysis.append([cluster_count, kmeans_model.inertia_])
    
    analysis_results = pd.DataFrame(cluster_analysis, columns=['cluster_count', 'inertia'])

    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4, style="whitegrid")
    sns.lineplot(data=analysis_results, x='cluster_count', y='inertia').set(
        title="Elbow Method for Optimal Clusters"
    )
    plt.show()


def execute_kmeans_clustering(scored_rfm, original_rfm, num_clusters=4):
    rfm_array = scored_rfm.values 
    clustering_model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300)
    cluster_labels = clustering_model.fit_predict(rfm_array)
    
    scored_rfm['cluster_group'] = cluster_labels
    original_rfm['cluster_group'] = cluster_labels
    final_rfm_data = original_rfm
    return final_rfm_data
    

def create_cluster_visualization(clustered_data):
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection='3d')
    
    cluster_groups = clustered_data.groupby('cluster_group')
    
    for cluster_id, (group_label, group_data) in enumerate(cluster_groups):
        x_values = group_data['Recency']
        y_values = group_data['MonetaryValue']
        z_values = group_data['Frequency']
        axis.scatter(x_values, y_values, z_values, s=50, alpha=0.6, 
                    edgecolors='w', label=f'Cluster {group_label}')
        
    axis.set_xlabel('Recency')
    axis.set_zlabel('Frequency')
    axis.set_ylabel('MonetaryValue')
    plt.title('3D Visualization of Customer Segments')
    plt.legend()
    plt.show()


def create_segmentation_report(clustered_data):
    cluster_labels = {0: 'departing_customers', 1: 'active_customers', 
                     2: 'inactive_customers', 3: 'new_customers'}
    clustered_data['CustomerSegment'] = clustered_data['cluster_group'].map(cluster_labels)
    
    print(clustered_data.head())
    
    print("\nCustomer segment distribution (%):")
    cluster_distribution = (clustered_data.cluster_group.value_counts(normalize=True, sort=True) * 100)
    print(cluster_distribution.to_string())
    
    print("\nOverall dataset statistics:")
    print(clustered_data.agg(['mean']))
    
    print("\nCluster size analysis:")
    clustered_data.cluster_group.value_counts().plot(
        kind='bar', figsize=(6, 4), 
        title='Customer Distribution Across Segments'
    )
    plt.show()
    
    clustered_data.to_csv(r'Segmentation_final\customer_segments_output.csv', index=False)  


if __name__ == '__main__':
    raw_data = load_and_preprocess_data(DATA_PATH)
    analysis_date = dt.datetime(2005, 5, 31)
    rfm_table = build_rfm_metrics(raw_data, analysis_date)
    rfm_scored = compute_rfm_scores(rfm_table)
    display_elbow_analysis(rfm_scored)
    segmented_data = execute_kmeans_clustering(rfm_scored, rfm_table)
    create_cluster_visualization(segmented_data)
    create_segmentation_report(segmented_data)
