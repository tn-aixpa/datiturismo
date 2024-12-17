from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd

class PCAConfig:
    def __init__(self, is_pca_enabled, num_components):
        self._is_pca_enabled = is_pca_enabled
        self._num_components = num_components   
        
    def is_enabled(self):
        return self._is_pca_enabled
        
    def get_nr_components(self):
        return self._num_components


class Clustering:
    def __init__(self, training_ds):
        self._training_ds = training_ds
        
    def apply_standardization(self, training_ds):
        # Standardizing data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(training_ds)
        return scaler, scaled_data
    
    def fit_model(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=1, init='k-means++')
        scaler, training_ds = apply_standardization(self._training_ds)
        predictions = kmeans.fit_predict(training_ds) 
        counts = Counter(kmeans.labels_)  
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=self._training_ds.columns) 
        centers['size'] = [counts[i] for i in range(n_clusters)] 
        print(centers.sort_values(by="size"))
        return scaler, kmeans, predictions
        
    def plot_cv_clusters(self, training_ds, n_clusters, ax):
        # fit the kmeans model
        scaler, kmeans, predictions = fit_model(training_ds, n_clusters)
        # plot the data according to the obtained clusters    
        for cluster in range(n_clusters):
            color = random.choice(list(mcolors.CSS4_COLORS.keys()))
            ax.scatter(training_ds.iloc[predictions==cluster, 0], training_ds.iloc[predictions==cluster, 1], label=cluster, color=color)
        # rescale centroids to their original unit
        rescaled_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(rescaled_centroids[:, 0], rescaled_centroids[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroids')
        plt.xlabel('Tourists') 
        plt.ylabel('Excursionists') 
        ax.legend(scatterpoints=1)
    
    def plot_elbow_method(self, cluster_range, data_scaled, ax):
        SSE = [] # inertia or distortion
        for cluster in range(2,cluster_range):
            kmeans = KMeans(n_clusters = cluster, init='k-means++')
            kmeans.fit(data_scaled)
            SSE.append(kmeans.inertia_)    
        # converting the results into a dataframe and plotting them
        frame = pd.DataFrame({'Cluster':range(2,cluster_range), 'SSE':SSE})
        plt.figure(figsize=(12,6))
        ax.plot(frame['Cluster'], frame['SSE'], marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')

        
def data_imputation_weather(nearby_weather, target_weather):
    """
    Filling in missing weather data for specific days
    """
    missing_dates = set(nearby_weather["data"]) - set(target_weather.index)
    comune_near_importance_order = ["MEZZOLOMBARDO", "PERGINE VALSUGANA", "TRENTO"]
    # format the categorical data 
    nearby_weather["comune"] = pd.Categorical(nearby_weather["comune"], categories = comune_near_importance_order)
    nearby_weather = nearby_weather.sort_values(by = "comune")
    filled_missing_dates = []
    for curr_date in missing_dates:    
        rows = nearby_weather[nearby_weather["data"] == curr_date]
        rows = rows.sort_values(by = "comune")
        filled_missing_dates.append({"data": rows.iloc[0, :].to_dict()["data"],
                            "probprec06-12": rows.iloc[0, :].to_dict()["probprec06-12"]})
    filled_missing_dates = pd.DataFrame(data=filled_missing_dates)
    return filled_missing_dates


def define_regressor(training_data_T_E, features, target, pca_config: PCAConfig):
    # Step 1: Train test split
    X = training_data_T_E.loc[:, features].values
    y = training_data_T_E.loc[:, target].values
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Step 2: Standardization
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    
    # Step 3: Apply PCA transformation
    if pca_config.is_enabled():
        pca = PCA(n_components=pca_config.get_nr_components())
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"PCA Explained variance : {pca.explained_variance_ratio_}")
    
    # Step 4: Fitting Linear Regression to the training set
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)
        
    # Step 5: Prediction and accuracy evaluation
    y_predictions = linear_regressor.predict(X_test)
    r2 = r2_score(y_test, y_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, y_predictions))
    mae = mean_absolute_error(y_test, y_predictions)
    print(f"Model Evaluation Metrics: r2_score: {r2},  root mean squared error: {rmse}, mean absolute error: {mae}")
 
    

    
