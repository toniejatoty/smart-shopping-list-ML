import pytorchmodel
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
pd.set_option('display.max_columns', 20)   

class ProductProcessor:
    def __init__(self):
        self.product_to_id = {}
        self.id_to_product = {}
        self.category_to_id = {}
        self.id_to_category = {}
        self.product_categories = {}  
        self.next_product_id = 1
        self.next_category_id = 1
    def process_data(self, users_data):
        for i in range(len(users_data)):
            users_data[i]['timestamp'] = (datetime.now() - datetime.strptime(users_data[i]['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")).days 

        for i in range(len(users_data)):
            for j in range(len(users_data[i]['products'])):
                if users_data[i]['products'][j] not in self.product_to_id:
                    self.product_to_id[users_data[i]['products'][j]] = self.next_product_id
                    self.id_to_product[self.next_product_id] = users_data[i]['products'][j]
                    self.next_product_id += 1
                users_data[i]['products'][j] = self.product_to_id[users_data[i]['products'][j]]
            for j in range(len(users_data[i]['categories'])):
                if users_data[i]['categories'][j] not in self.category_to_id:
                    self.category_to_id[users_data[i]['categories'][j]] = self.next_category_id
                    self.id_to_category[self.next_category_id] = users_data[i]['categories'][j]
                    self.next_category_id += 1
                users_data[i]['categories'][j] = self.category_to_id[users_data[i]['categories'][j]]
        
        
        return users_data
    def get_vocab_size(self):
        return len(self.product_to_id)
    
    def get_num_categories(self):
        return len(self.category_to_id)


BASE_DIR = Path(__file__).resolve().parent  
data_dir = BASE_DIR / "data"
file_path = data_dir / "example_input.json"

data = json.load(open(file_path, 'r', encoding='utf-8'))
productprocessor = ProductProcessor()
user_data = productprocessor.process_data(data['users_data'])
user_data_df = pd.DataFrame(user_data) 

print(user_data_df)
#print(user_data_df['user_id'].value_counts())
pytorchmodel.get_prediction(user_data_df, productprocessor)