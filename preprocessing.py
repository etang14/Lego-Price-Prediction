import pandas as pd
import numpy as np

from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def merge_data(file1, file2):
    """
    Script to merge two Lego data files and return a pandas dataframe while also writing results
    to a new .csv
    """
    df1 = pd.read_csv(file1)
    df1.dropna(subset=['USD_MSRP', 'Current_Price'], inplace=True)
    df1.drop_duplicates(["Set_ID"], inplace=True)

    df2 = pd.read_excel(file2, sheet_name="Sheet1")
    df2.info()
    df2 = df2[["US_retailPrice", "number", "numberVariant", "lastUpdated", "ownedBy", "wantedBy"]]
    df2.dropna(subset=['number', 'US_retailPrice'], inplace=True)
    df2["Set_ID"] = df2['number'].astype(str) + '-' + df2['numberVariant'].astype(str)
    df2.drop_duplicates(["Set_ID"], inplace=True)

    res_df = pd.merge(df1, df2, on=["Set_ID"], how="outer")
    res_df.info()
    
    res_df = res_df[["Set_ID", "Name", "Year", "Theme", "Theme_Group", "Subtheme", "Category", "Packaging", 
                     "Num_Instructions", "Availability", "Pieces", "Minifigures", "Owned", "Rating", 
                     "USD_MSRP", "Total_Quantity", "Current_Price", "lastUpdated", "US_retailPrice",
                     "ownedBy", "wantedBy"]]

    res_df.dropna(subset=['USD_MSRP', 'Current_Price'], inplace=True)
    res_df.to_csv("LEGO_Data/merged.csv")
    return res_df

def get_data(filename, years, xgb_=True):
    """
    Reads a .csv of Lego data, cleans and preprocesses data for training and visualization.
    """
    df = pd.read_csv(filename)
    df.drop(["lastUpdated", "Owned"], axis=1, inplace=True)
    # drop sets without price data and ratings
    df.drop_duplicates(inplace=True)
    df = df[df['Rating']>0]

    # optional bounds for years under consideration
    df = df[df['Year']>years[0]]
    df = df[df['Year']<=years[1]]

    # re-encode years
    label_encoder = LabelEncoder()
    df["Year"]= label_encoder.fit_transform(df["Year"])
    
    # relabel Packaging that isn't specified to prevent collision with same label in Availability
    df["Packaging"] = df["Packaging"].replace("{Not specified}", "Unknown")
    df.astype({"Packaging": "object"})

    # one-hot encode features that have low cardinality
    categories = ["Category", "Packaging", "Availability", "Theme_Group"]
    for cat in categories:
        one_hot = pd.get_dummies(df[cat], dtype='int')
        df = df.drop(cat, axis=1)
        df = pd.concat([df, one_hot], axis=1)

    # fill all empty entries with 0 (just for minifigure counts)
    df.fillna(0, inplace=True)

    df = df[np.abs(stats.zscore(df['Pieces'])) < 3].reset_index(drop=True)
    df = df[np.abs(stats.zscore(df['USD_MSRP'])) < 3].reset_index(drop=True)
    
    if xgb_:
        # Theme Group and Subtheme quite similar to Theme with no real effect on train/test performance
        # df["Theme_Group"] = pd.Categorical(df["Theme_Group"])
        # df["Subtheme"] = pd.Categorical(df["Subtheme"])
        df["Theme"] = pd.Categorical(df["Theme"])
    
        X = df.drop(["Current_Price", "Set_ID", "Name", "Subtheme", "US_retailPrice"], axis=1)
    else:
        X = df.drop(["Current_Price", "Set_ID", "Name", "Subtheme", "Theme", "US_retailPrice"], axis=1)
    
    #X = df[["Pieces", "Rating", "USD_MSRP", "Total_Quantity", "Year"]]
    
    y = df['Current_Price']

    return X, y

    
