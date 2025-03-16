import requests
import pandas as pd

country_code = "all"
year_range = "2000:2023"
indicators = {
    "GDP (USD)": "NY.GDP.MKTP.CD",
    "Inflation Rate (%)": "FP.CPI.TOTL.ZG",
    "GNI per capita (USD)": "NY.GNP.PCAP.CD",
    "Unemployment Rate (%)": "SL.UEM.TOTL.ZS",
    "Population": "SP.POP.TOTL"
}
df = pd.DataFrame()

for indicator_name, indicator_code in indicators.items():
    
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=20000"
    response = requests.get(url)
    data = response.json()
    
    if data and isinstance(data, list) and len(data) > 1:
        records = data[1]
        
        temp_df = pd.DataFrame(records)[["country", "date", "value"]]
        temp_df["Country"] = temp_df["country"].apply(lambda x: x["value"])
        temp_df["Year"] = temp_df["date"].astype(int)
        temp_df = temp_df.drop(columns=["country", "date"])
        
        temp_df.rename(columns={"value": indicator_name}, inplace=True)
        
        if df.empty:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on=["Country", "Year"], how="outer")

df.to_csv("world_bank_dataset.csv", index=False)