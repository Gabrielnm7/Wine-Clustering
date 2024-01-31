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
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
url = "https://storage.googleapis.com/the_public_bucket/wine-clustering.csv"
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="."), name="static")

data_wine = pd.read_csv(url)

@app.get("/")
def introduction():
    # Project presentation
    project_title = "<h1>Wine Clustering Project</h1>"
    project_introduction = "<p>The Wine Clustering Project endeavors to explore and analyze a dataset encompassing various attributes of different types of wines and discover some patterns. First, let's dive into the dataset to see how it is.</p>"

    # checking data shape
    row, col = data_wine.shape
    html_content = f'<p>There are {row} rows and {col} columns</p>'
    html_content += "<p>Let's see the first 5 rows:</p>"
    html_content += data_wine.head(5).to_html()

    # Scaling the data to keep the different attributes in same range.
    data_normalized = (data_wine - data_wine.min())/(data_wine.max()-data_wine.min())

    # Convert DataFrame to HTML format
    html_content += "<h2>Normalized data:</h2>"
    html_content += "<p>Although there are few outliers, the attributes exhibit varying scales. Normalization ensures consistency in scale across all features, enhancing the effectiveness of subsequent analysis</p>"
    html_content += data_normalized.head(5).to_html()

    # Combine project presentation with dataset information
    full_html_content = project_title + project_introduction + html_content

    return HTMLResponse(content=full_html_content)


def clustering_analysis():
    data_normalized = (data_wine - data_wine.min()) / (data_wine.max() - data_wine.min())
    # WCSS For elbow method
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i,max_iter=400)
        kmeans.fit(data_normalized)
        wcss.append(kmeans.inertia_)
    #Elbow graph
    plt.figure(figsize=(12, 8))
    plt.plot(range(1,11),wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.tight_layout()
    plt.savefig('Elbow_Method.png')    

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
    plt.figure(figsize=(10, 6))
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

@app.get("/Analysis")
def analysis(request: Request):
    # EDA
    html_content = "<h1>Exploratory Data Analysis</h1>"
    html_content += "<h3> We first use the .describe() function to generate descriptive statistics for numerical attributes, and then calculate the correlation matrix using the .corr() function </h3>"
    html_content += data_wine.describe().to_html()
    html_content += "<p> Now, the correlation matrix \n </p>"
    maps = data_wine.corr()
    html_content += maps.style.background_gradient(cmap="coolwarm").to_html()
    html_content += """\n <h4>
    Displaying the correlation matrix provides us with valuable information about the relationships between different attributes. 
    Visualizing the correlation matrix using a heatmap enhances our understanding by highlighting strong and weak correlations. 
    As we can see, <b>Flavanoids and Total_Phenols</b> exhibit strong correlations with nearly all other variables in the dataset. 
    This could be suggesting the existing linear relationship between these attributes and others 
    it's worth highlighting the noteworthy correlations:
<ul>
    <li>Flavanoids and OD280 <span style="font-weight: bold;">(0.78)</span></li>
    <li>Total_Phenols and OD280 <span style="font-weight: bold;">(0.699)</span></li>
    <li>Alcohol and Proline <span style="font-weight: bold;">(0.64)</span></li>
</ul>

These correlations suggest potential associations that may merit closer examination.</h4> 
<h3>Let's visualize these relationships using scatter plots </h3>"""
    # Scatter plots
    def create_scatterplot(x, y, xlabel, ylabel, title):
        plt.figure()
        sns.scatterplot(x=x, y=y, data=data_wine)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(f"{x+'vs'+y}.png")

    create_scatterplot('Flavanoids', 'Total_Phenols', 'Flavanoids', 'Total Phenols', 'Scatter Plot: Flavanoids vs Total Phenols')
    create_scatterplot('Flavanoids', 'OD280', 'Flavanoids', 'OD280', 'Scatter Plot: Flavanoids vs OD280')
    create_scatterplot('Total_Phenols', 'OD280', 'Total Phenols', 'OD280', 'Scatter Plot: Total Phenols vs OD280')
    create_scatterplot('Flavanoids','Proanthocyanins','Flavanoids', 'Proanthocyanins', 'Scatter Plot: Flavanoids vs Proanthocyanins')
    create_scatterplot('Alcohol', 'Proline', 'Alcohol', 'Proline', 'Scatter Plot: Alcohol vs Proline')
    html_content += """
    <img src="/static/FlavanoidsvsOD280.png" alt="Boxplot">
    <img src="/static/FlavanoidsvsProanthocyanins.png" alt="Histogram">
    <img src="/static/FlavanoidsvsTotal_Phenols.png" alt="PCA Plot">
    <img src="/static/AlcoholvsProline.png" alt="PCA Plot">
    <img src="/static/Total_PhenolsvsOD280.png" alt="PCA Plot">
    
    """

    return HTMLResponse(content=html_content)

@app.get("/Clustering")
def pca_clustering():
    clustering_analysis()

    html_content = """
    <h1 center>Clustering Analysis</h1>
    <h3> Determine Optimal Number of Clusters </h3>
    <p>First, we calculate the Within-Cluster Sum of Squares (WCSS) using the Elbow Method to identify potential cluster counts. 
    Then, we validate our choice using the silhouette score to ensure robustness in our clustering approach. </p>
    <img src="/static/Elbow_Method.png">
    <h3>It appears that three clusters provide a suitable choice for our KMeans algorithm.
    Therefore, we will proceed with clustering using three clusters and a maximum iteration limit of 400. \n
    Let's look at their differences: </h3>
    <h3>Boxplot</h3>
    <img src="/static/boxplot.png">
    <h3>Histogram</h3>
    <img src="/static/histogram.png">

    """
    html_content += """
    <h4>In examining the clusters, notable distinctions emerge, particularly between clusters 0 and 2. 
    Cluster 0 exhibits a markedly higher mean magnesium level, averaging approximately $107.87 mg/L$, and a notably elevated mean proline content, averaging around $1117.82 mg/L$.</h4>

    <h4>In contrast, Cluster 2 displays a relatively lower magnesium level, averaging around $92.95 mg/L$, and a mean proline content lower than Cluster 0 but higher than Cluster 1, averaging approximately $500.17 mg/L$.</h4>

    <h4>Furthermore, Cluster 0 boasts the highest alcohol content among all clusters, while Cluster 2 exhibits the lowest.</h4>

    <h4>These findings suggest that Cluster 0 shares characteristics consistent with red wine, characterized by its higher magnesium and proline levels and alcohol content, whereas Cluster 2 aligns more closely with white wine, marked by comparatively lower magnesium and proline levels and alcohol content.</h4>
    """

    html_content += "<h2>Now, let's take a look by performin PCA and see the differents cluster in 2 dimensions:</h2>"
    html_content += """
    <h3>PCA Plot</h3>
    <img src="/static/pca_plot.png">"""

    html_content += """<h2>Conclusion</h2>
    <h3>I conclude that the clustering analysis effectively partitioned the dataset into meaningful clusters, providing valuable insights into the underlying structure of the data. 
    Further analysis and interpretation may be conducted to explore the characteristics and patterns within each cluster</h3>"""
    return HTMLResponse(content=html_content)

if __name__=="__main__":
    uvicorn.run(app,port=8000,host="0.0.0.0")