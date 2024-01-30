import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.simplefilter("ignore")
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
url = "https://storage.googleapis.com/the_public_bucket/wine-clustering.csv"

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def load_dataset():
    """
    Loading the dataset in pandas dataframe
    return: two datasets, the first one is the dataframe and the 2nd is the same 
    but normalized. 
    """
    # loading wine dataset
    data_wine = pd.read_csv(url)

    # checking data shape
    row, col = data_wine.shape
    html_content = f'<p>There are {row} rows and {col} columns</p>'
    html_content += "<p>Let's see the first 5 rows:</p>"
    html_content += data_wine.head(5).to_html()

    # Scaling the data to keep the different attributes in same range.
    data_normalized = (data_wine - data_wine.min())/(data_wine.max()-data_wine.min())

    # Convert DataFrame to HTML format
    html_content += "<p>Normalized data:</p>"
    html_content += data_normalized.head(5).to_html()

    return HTMLResponse(content=html_content)

def clustering_analysis():
    data_wine = pd.read_csv(url)
    data_normalized = (data_wine - data_wine.min()) / (data_wine.max() - data_wine.min())

    clustering = KMeans(n_clusters=3, max_iter=300)
    clustering.fit(data_normalized)
    data_normalized["KMeans_Clusters"] = clustering.labels_
    data_wine["KMeans_Clusters"] = clustering.labels_

    # Boxplot
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(data_wine.columns[:-1]):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x='KMeans_Clusters', y=feature, data=data_wine)
        plt.title(feature)
        plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig('boxplot.png')

    # Histogram
    plt.figure(figsize=(13, 9))
    for i, feature in enumerate(data_wine.columns[:-1]):
        plt.subplot(3, 5, i + 1)
        for cluster in range(3):
            sns.histplot(data_wine[data_wine['KMeans_Clusters'] == cluster][feature], kde=True, label=f'Cluster {cluster}')
        plt.title(feature)
        plt.legend()
    plt.tight_layout()
    plt.savefig('histogram.png')

    # PCA
    pca = PCA(n_components=2)
    pca_wines = pca.fit_transform(data_normalized.drop(columns=['KMeans_Clusters']))
    pca_wines_df = pd.DataFrame(data=pca_wines, columns=["Component_1", "Component_2"])
    pca_name_wines = pd.concat([pca_wines_df, data_normalized[["KMeans_Clusters"]]], axis=1)

    X = data_normalized.drop(columns=['KMeans_Clusters'])
    y = data_normalized['KMeans_Clusters']

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for cluster_label in y.unique():
        plt.scatter(X_pca[y == cluster_label, 0], X_pca[y == cluster_label, 1], label=f'Cluster {cluster_label}')

    plt.legend(title='Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Plot of Wine Data with Clusters')
    plt.savefig('pca_plot.png')

@app.get("/Clustering")
def pca_clustering():
    clustering_analysis()

    html_content = """
    <h2>Clustering Analysis</h2>
    <h3>Boxplot</h3>
    <img src="/static/boxplot.png" alt="Boxplot">
    <h3>Histogram</h3>
    <img src="/static/histogram.png" alt="Histogram">
    <h3>PCA Plot</h3>
    <img src="/static/pca_plot.png" alt="PCA Plot">
    
    """

    return HTMLResponse(content=html_content)

if __name__=="__main__":
    uvicorn.run(app,port=8000,host="192.168.0.175")
