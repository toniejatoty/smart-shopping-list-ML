import pandas as pd
from pathlib import Path
import ast
import pickle
pd.set_option('display.max_columns', 20)   

class ProductProcessor:
    def __init__(self):
        self.product_to_id = {}
        self.id_to_product = {}
        self.category_to_id = {}
        self.id_to_category = {}
        self.product_categories = {}  
        self.next_off_product_id = 1
        self.next_category_id = 1
        self.max_basket_len = 0
        self.max_cats_per_product = 0
        
        self.original_id_to_name = {}
        self._load_product_names()
    
    def _load_product_names(self):
        try:
            project_root = Path(__file__).parent.parent
            products_file = project_root / 'prepare_data_for_model' / 'OpenFoodData.csv'
            products_df = pd.read_csv(products_file)
            
            for _, row in products_df.iterrows():
                self.original_id_to_name[row['Id']] = row['Name']
            
            print(f"✅ Załadowano nazwy {len(self.original_id_to_name):,} produktów")
            
            sample_ids = list(self.original_id_to_name.keys())[:5]
            print("Przykładowe mapowania:")
            for pid in sample_ids:
                print(f"  {pid} -> {self.original_id_to_name[pid][:30]}...")
                
        except Exception as e:
            print(f"⚠️ Nie udało się załadować nazw produktów: {e}")
            print("Będę używać tylko ID")
    
    def _map_and_replace(self, data_series, to_id_dict, id_to_dict, next_id, is_product=True):
        
        all_values = set()
        
        for row in data_series:
            if isinstance(row, list):
                if is_product:
                    self.max_basket_len = max(self.max_basket_len, len(row))
                for item in row:
                    if isinstance(item, list) and not is_product:
                        self.max_cats_per_product = max(self.max_cats_per_product, len(item))
                        for subitem in item:
                            if subitem is not None and str(subitem) != 'nan':
                                all_values.add(str(subitem))
                        
                    elif item is not None and str(item) != 'nan':
                        #dla produktów
                        all_values.add(str(item))
        
        for value in all_values:
            if value not in to_id_dict:
                to_id_dict[value] = next_id
                id_to_dict[next_id] = value
                next_id += 1
        
        if is_product:
            def map_products(product_list, mapping_dict):
                if not isinstance(product_list, list):
                    return []
                
                result = []
                for item in product_list:
                    if item is not None and str(item) != 'nan':
                        mapped_id = mapping_dict.get(str(item), 0)
                        if mapped_id > 0:
                            result.append(mapped_id)
                
                return result
            
            mapped_series = data_series.apply(
                lambda x: map_products(x, to_id_dict)
            )
        
        else:
            # DLA KATEGORII: lista list ID
            def map_list_of_lists(nested_list, mapping_dict):
                """Mapuj listę list na listę list ID"""
                if not isinstance(nested_list, list):
                    return []
                
                result = []
                for sublist in nested_list:
                    if isinstance(sublist, list):
                        mapped_sublist = [mapping_dict.get(str(item), 0) for item in sublist]
                        result.append(mapped_sublist)
                    elif sublist is not None:
                        result.append([mapping_dict.get(str(sublist), 0)])
                    else:
                        result.append([0])
                
                return result
            
            mapped_series = data_series.apply(
                lambda x: map_list_of_lists(x, to_id_dict)
            )
        
        return mapped_series, next_id
        
    def process_data(self, users_data):
        users_data['off_product_id'], self.next_off_product_id = self._map_and_replace(
            users_data['off_product_id'], 
            self.product_to_id, 
            self.id_to_product, 
            self.next_off_product_id,
            is_product=True
        )
        
        users_data['aisle_id'], self.next_category_id = self._map_and_replace(
            users_data['aisle_id'], 
            self.category_to_id, 
            self.id_to_category, 
            self.next_category_id,
            is_product=False
        )
        
        return users_data
    
    def get_product_name(self, off_product_id):
        original_id = self.id_to_product[off_product_id]
        return self.original_id_to_name[original_id]
    
    def get_vocab_size(self):
        return len(self.product_to_id)
    
    def get_num_categories(self):
        return len(self.category_to_id)
    
    def save_processed_data(self, data_df, output_dir):
        output_dir = Path(output_dir)
        
        data_path = output_dir / 'processed_data.pkl'
        data_df.to_pickle(data_path)
        print(f"✅ Zapisano dane: {data_path}")
        
        mappings = {
            'product_to_id': self.product_to_id,
            'id_to_product': self.id_to_product,
            'category_to_id': self.category_to_id,
            'id_to_category': self.id_to_category,
            'product_categories': self.product_categories,
            'original_id_to_name': self.original_id_to_name,
            'next_off_product_id': self.next_off_product_id,
            'next_category_id': self.next_category_id,
            'max_basket_len': self.max_basket_len,
            'max_cats_per_product': self.max_cats_per_product
        }
        
        mappings_path = output_dir / 'mappings.pkl'
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)
        print(f"✅ Zapisano mapowania: {mappings_path}")
        
        csv_path = output_dir / 'processed_data.csv'
        data_df.to_csv(csv_path, index=False)
        print(f"✅ Zapisano CSV: {csv_path}")
        
        print(f"\n📦 Zapisano wszystko w: {output_dir}")
    
    def load_processed_data(self, input_dir):
        input_dir = Path(input_dir)
        
        mappings_path = input_dir / 'mappings.pkl'
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        self.product_to_id = mappings['product_to_id']
        self.id_to_product = mappings['id_to_product']
        self.category_to_id = mappings['category_to_id']
        self.id_to_category = mappings['id_to_category']
        self.product_categories = mappings['product_categories']
        self.original_id_to_name = mappings['original_id_to_name']
        self.next_off_product_id = mappings['next_off_product_id']
        self.next_category_id = mappings['next_category_id']
        self.max_basket_len = mappings['max_basket_len']
        self.max_cats_per_product = mappings['max_cats_per_product']
        
        print(f"✅ Wczytano mapowania z: {mappings_path}")
        
        # 2. Wczytaj DataFrame
        data_path = input_dir / 'processed_data.pkl'
        data_df = pd.read_pickle(data_path)
        print(f"✅ Wczytano dane: {data_path}")
        
        return data_df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent  
    data_dir = BASE_DIR / "data"
    processed_dir = data_dir / "processed"  
    file_path = data_dir / "kaggle_prepared.csv"
    
    
    def safe_literal_eval(x):
        # changes '[1,2,3]' into [1,2,3] string-> list
        if pd.isna(x):
            return []
        x_str = str(x).strip()
        if not x_str or x_str == '[]' or x_str == 'nan':
            print('pusta lista')
            return []
        if x_str.startswith('[') and x_str.endswith(']'):
            try:
                return ast.literal_eval(x_str)
            except (SyntaxError, ValueError):
                x_str = x_str[1:-1]
                if ',' in x_str:
                    items = [item.strip().strip("'\"") for item in x_str.split(',')]
                    return items
                elif x_str:
                    return [x_str]
                else:
                    return []
        else:
            return [x_str]

    productprocessor = ProductProcessor()
    
    print("=" * 60)
    print("PRZETWARZANIE DANYCH OD NOWA")
    print("=" * 60)
    
    data = pd.read_csv(file_path, converters={
        'off_product_id': safe_literal_eval,
        'aisle_id': safe_literal_eval
    })
    
    user_data = productprocessor.process_data(data)
    user_data_df = pd.DataFrame(user_data) 
    
    print(f"\n📊 STATYSTYKI:")
    print(f"  Maksymalna liczba produktów w koszyku: {productprocessor.max_basket_len}")
    print(f"  Maksymalna liczba kategorii per produkt: {productprocessor.max_cats_per_product}")
    print(f"\n✅ Przetworzono {len(user_data_df):,} wierszy")
    print(f"✅ Unikalnych produktów: {productprocessor.get_vocab_size():,}")
    print(f"✅ Unikalnych kategorii: {productprocessor.get_num_categories():,}")
    
    productprocessor.save_processed_data(user_data_df, processed_dir)
    