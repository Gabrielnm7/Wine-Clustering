## <h1 align ="center"> Wine Clusterin Project </h1>

The *Wine Clustering Project* endeavors to explore and analyze a dataset encompassing various attributes of different types of wines. 

By employing data analysis techniques and unsupervised learning algorithms such as KMeans clustering and Principal Component Analysis (PCA), the project seeks to uncover hidden patterns and structures within the dataset. 

Some curious pattern to see is the potential positive linear relationship between attributes like flavanoids and total_phenols:
<p>
<img src="./_src/images/linear_relationship.png"  height=345>
</p>

 Through this analysis, we aim to gain insights into the characteristics of different types of wines and potentially identify distinct clusters based on their attributes. At the end, we perform a KMeans Algorithm to cluster the different types, resulting in clear visualizations like the PCA plots shown below

<p>
<img src="./_src/images/PCA_2dims.png" height=325>
<img src="./_src/images/pca_3dims.png" height=375> 

As demonstrated, the clustering is well done, offering valuable insights into the underlying structure of the data


## **Dataset**
The Wine dataset is available for download [here](https://storage.googleapis.com/the_public_bucket/wine-clustering.csv)

## **Analysis Details**
### For a detailed analysis, you can refer to the EDA and Unsupervised Learning notebooks by pressing [here](https://github.com/Gabrielnm7/Wine-Clustering/blob/main/EDA.ipynb)

## Environment 
<pre><code>Python 3.9.5 
Numpy 1.23.0
Pandas 1.4.3
Matplotlib 3.6.2
Seaborn 0.12.2
Scikit Learn 1.2.2
Uvicorn 0.22.0
Fastapi 0.99.1
</code></pre>
# <center >FastAPI
### In this stage of the project, the analysis of the wine dataset is made available using the FastAPI framework. Two functions have been defined for the endpoints that will be consumed in the API, each with a @app.get('/') decorator. Below are the functions and the queries that can be performed through the API:

- **"/"**: This endpoint provides access to the dataset and its normalization. It returns a summary of the dataset along with its normalized version.

- **"/Analysis"**: Accessing this endpoint allows users to see exploratory data analysis (EDA) on the dataset. It provides descriptive statistics such as mean, median, and quartiles, as well as a correlation matrix visualization.

- **"/Clustering"**: This endpoint offers clustering analysis of the dataset using the K-means algorithm. It clusters the data points into distinct groups based on their features, facilitating further insights into patterns and relationships within the data.

# **Docker Setup Documentation**
### 1. **Navigate to the project directory:**

- Open a terminal or command prompt and navigate to the directory where your project files are located. For example, if your files are in **C:\Users\admin\Documents\GitHub\Wine-Clustering** , you should navigate to that directory in your terminal.

### 2. **Build the Docker image:**
- Run the following command to build the Docker image:
> docker build -t your_image_name .

- Replace *"your_image_name"* with the name you want to give your Docker image. The dot at the end of the command indicates that the Dockerfile is located in the current directory.

### 3. **Wait for the build to complete:**

- Docker will download the base Python 3.9.5 image if it's not already available locally and then run the steps defined in your Dockerfile to build the image. Wait for this process to complete.

### 4. Run the Docker container:

- Once the Docker image has been successfully built, you can run a container based on that image. Use the following command:
> docker run -p local_port:8000 your_image_name
- Replace *"local_port"* with the port number on your local machine where you want the container to expose your application. For example, if you want the application to be available on port 8000 of your local machine, use "8000". 
- Replace *"your_image_name"* with the name of the image you specified in the previous step.

### 5. Access your application:

- Once the container is up and running, you can access the pipeline through a web browser. If you followed the example above and exposed your application on port 8000, you can access it by navigating to http://localhost:8000 in your web browser. 

