import pandas as pd

class PCAConfig:
    def __init__(self, is_pca_enabled, num_components):
        self._is_pca_enabled = is_pca_enabled
        self._num_components = num_components   
        
    def is_enabled(self):
        return self._is_pca_enabled
        
    def get_nr_components(self):
        return self._num_components

        
def data_imputation_weather(nearby_weather, target_weather):
    """
    Filling in missing weather data for specific  days
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
 
    
    
    
    
