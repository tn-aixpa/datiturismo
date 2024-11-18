import pandas as pd

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
    
    
    
    
