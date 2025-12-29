import pandas as pd
import numpy as np

# preprocssing
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return d

class Prep:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(self.filename).drop("Delivery_person_ID", axis=1)
        
    def clean(self):
        df = self.df
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
        
        # in "Time_Orderd" and Time_Order_picked column, there are multiple invalid times which we will drop 
        # 1. force columns to string
        df["Time_Orderd"] = df["Time_Orderd"].astype(str)
        df["Time_Order_picked"] = df["Time_Order_picked"].astype(str)
        
        # 2. remove invalid 24:xx times
        df = df[~df["Time_Orderd"].str.startswith("24:")]
        df = df[~df["Time_Order_picked"].str.startswith("24:")]
        
        # 3. keep only rows containing a colon (:)
        df = df[df["Time_Orderd"].str.contains(":", na=False)]
        df = df[df["Time_Order_picked"].str.contains(":", na=False)]
        
        # 4. trim seconds (HH:MM:SS → HH:MM)
        df["Time_Orderd"] = df["Time_Orderd"].str.slice(0, 5)
        df["Time_Order_picked"] = df["Time_Order_picked"].str.slice(0, 5)
        
        # 5. convert to datetime
        df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M").dt.time
        df["Time_Order_picked"] = pd.to_datetime(df["Time_Order_picked"], format="%H:%M").dt.time
        
        df["distance_miles"] = haversine_miles(
            df["Restaurant_latitude"],
            df["Restaurant_longitude"],
            df["Delivery_location_latitude"],
            df["Delivery_location_longitude"]
        )
        
        df = df.dropna()
        df = df[df["distance_miles"] < 1000]
        return(df)




