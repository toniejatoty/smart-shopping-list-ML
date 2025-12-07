import json
import os
from datetime import datetime
from config import supabase
import pandas as pd

def fetch_data_from_table(table_name):
    try:
        print(f"Downloading data from {table_name}")
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            query = supabase.table(table_name).select("*").range(offset, offset + page_size - 1)
            response = query.execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            print(f"  Downloaded {len(response.data)} records (sum: {len(all_data)})")
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
        
        print(f"Downloaded all {len(all_data)} records")
        return all_data
    except Exception as e:
        print(f"Error {e}")
        return None

def save_to_csv(data, filename):
    try:
        df = pd.DataFrame(data)
        full_path = os.path.join(os.getcwd(), filename)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"Data saved in: {full_path}")
        return full_path
    except Exception as e:
        print(f"Error with saving: {e}")
        return None

def export_table_to_csv(table_name, custom_filename):    
    data = fetch_data_from_table(table_name)
    
    if data:
        print(f"Saving {len(data)} records")
        
        if custom_filename:
            filename = custom_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{table_name}_export_{timestamp}.csv"
        
        save_to_csv(data, filename)
    else:
        print("Error")

export_table_to_csv("Products", 'OpenFoodData.csv')