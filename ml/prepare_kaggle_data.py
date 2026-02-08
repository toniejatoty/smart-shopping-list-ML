import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

BASE_DIR = Path(__file__).parent
KAGGLE_DIR = BASE_DIR / "data" / "kaggle" / "archive"
OFF_FILE = BASE_DIR / "data" / "OpenFoodData.csv"
MODEL_DIR = BASE_DIR.parent.parent / "product_category_model"
OUTPUT_FILE = BASE_DIR / "data" / "kaggle_prepared.csv"

def map_products(kaggle_products, off_df, model):
    off_filtered = off_df[off_df['Id'].notna() & off_df['Name'].notna()].copy()
    off_filtered['id'] = off_filtered['Id'].astype(str)
    off_filtered['name'] = off_filtered['Name'].astype(str).str.lower().str.strip()
    off_filtered['original_name'] = off_filtered['Name'].astype(str)
    
    off_products = off_filtered[['id', 'name', 'original_name']].to_dict('records')
    
    off_names = [p['original_name'] for p in off_products]
    off_embeddings = model.encode(off_names, show_progress_bar=True)
    off_embeddings_norm = off_embeddings / np.linalg.norm(off_embeddings, axis=1, keepdims=True)
    
    product_mapping = {}
    batch_size = 1000
    
    num_batches = (len(kaggle_products) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(kaggle_products), batch_size), total=num_batches, desc="Batch'e produktów"):
        batch = kaggle_products.iloc[i:i+batch_size]
        
        kaggle_names = batch['product_name'].astype(str).tolist()
        kaggle_embeddings = model.encode(kaggle_names, show_progress_bar=False)
        kaggle_embeddings_norm = kaggle_embeddings / np.linalg.norm(kaggle_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(kaggle_embeddings_norm, off_embeddings_norm.T)
        
        for idx, (_, row) in enumerate(batch.iterrows()):
            best_idx = np.argmax(similarities[idx])
            best_score = similarities[idx, best_idx]
            
            if best_score > 0.5:
                product_mapping[row['product_id']] = off_products[best_idx]['id']
    return product_mapping
    

def main():
    products = pd.read_csv(KAGGLE_DIR / "products.csv")
    prior = pd.read_csv(KAGGLE_DIR / "order_products__prior.csv")
    train = pd.read_csv(KAGGLE_DIR / "order_products__train.csv")
    orders = pd.read_csv(KAGGLE_DIR / "orders.csv")
    off_df = pd.read_csv(OFF_FILE)
    
    model = SentenceTransformer(str(MODEL_DIR))
    
    product_mapping = map_products(products, off_df, model)
    
    prior['off_product_id'] = prior['product_id'].map(product_mapping)
    train['off_product_id'] = train['product_id'].map(product_mapping)
    
    prior = prior[prior['off_product_id'].notna()][['order_id', 'off_product_id']]
    train = train[train['off_product_id'].notna()][['order_id', 'off_product_id']]
    
    prior = prior.groupby('order_id')['off_product_id'].apply(list).reset_index()
    train = train.groupby('order_id')['off_product_id'].apply(list).reset_index()
    
    orders = orders[orders['eval_set'] != 'test']
    orders = orders[['order_id', 'user_id', 'days_since_prior_order', 'order_number']]
    orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
    
    orders = pd.merge(orders, prior, on='order_id', how='left')
    orders = pd.merge(orders, train, on='order_id', how='left', suffixes=('_prior', '_train'))
    
    orders['off_product_id'] = orders['off_product_id_prior'].fillna(orders['off_product_id_train'])
    orders = orders.drop(['off_product_id_prior', 'off_product_id_train'], axis=1)
    
    orders = orders.sort_values(by=['user_id', 'order_number'], ascending=True)
    orders = orders.drop('order_number', axis=1)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
