import pandas as pd
import ast
from tqdm import tqdm

def map_products_and_categories(kaggle_processed, off_data):
    product_map = {}
    category_map = {}
    mapped_products = []
    mapped_categories = []
    
    off_dict = {}
    for _, row in tqdm(off_data.iterrows(), total=len(off_data), desc="Budowanie słownika"):
        raw_cats = row['Categories']
        if pd.isna(raw_cats):
            cats = []
        else:
            try:
                cats = ast.literal_eval(raw_cats) if isinstance(raw_cats, str) else []
            except:
                cats = []
        off_dict[str(row['Id'])] = cats

    for products_raw in tqdm(kaggle_processed['off_product_id'], desc="Mapowanie"):
        if pd.isna(products_raw) or not isinstance(products_raw, str):
            order_products = []
        else:
            try:
                order_products = ast.literal_eval(products_raw)
            except:
                order_products = []
        
        row_products_mapped = []
        row_categories_mapped = []
        
        for p_id in order_products:
            p_str = str(p_id)
            
            if p_str not in product_map:
                product_map[p_str] = len(product_map) + 1
            row_products_mapped.append(product_map[p_str])
            
            p_cats = off_dict.get(p_str, [])
            p_cats_mapped = []
            for c_name in p_cats:
                if c_name not in category_map:
                    category_map[c_name] = len(category_map) + 1
                p_cats_mapped.append(category_map[c_name])
            
            row_categories_mapped.append(p_cats_mapped)
            
        mapped_products.append(row_products_mapped)
        mapped_categories.append(row_categories_mapped)
        
    return mapped_products, mapped_categories, product_map, category_map

def make_model_input_dataframe(kaggle_processed, mapped_products, mapped_categories):

    final_data = pd.DataFrame({
        'user_id': kaggle_processed['user_id'],
        'order_id': kaggle_processed['order_id'],
        'days_since_prior_order': kaggle_processed['days_since_prior_order'].fillna(0),
        'off_product_id': mapped_products,
        'mapped_categories': mapped_categories
    })
    
    final_data = final_data.sort_values(['user_id', 'order_id']).reset_index(drop=True)
    
    return final_data