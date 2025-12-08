import pytorchmodel
import pandas as pd
from pathlib import Path
import ast
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
        
        # NOWE: Mapowanie oryginalnych ID do nazw produktów
        self.original_id_to_name = {}
        self._load_product_names()
    
    def _load_product_names(self):
        """Załaduj mapowanie ID produktu -> nazwa z pliku products.csv"""
        try:
            products_file = Path(__file__).parent / "kaggle" / "archive" / "products.csv"
            products_df = pd.read_csv(products_file)
            
            # Stwórz mapowanie: product_id -> product_name
            for _, row in products_df.iterrows():
                self.original_id_to_name[row['product_id']] = row['product_name']
            
            print(f"✅ Załadowano nazwy {len(self.original_id_to_name):,} produktów")
            
            # Sprawdź przykładowe mapowanie
            sample_ids = list(self.original_id_to_name.keys())[:5]
            print("Przykładowe mapowania:")
            for pid in sample_ids:
                print(f"  {pid} -> {self.original_id_to_name[pid][:30]}...")
                
        except Exception as e:
            print(f"⚠️ Nie udało się załadować nazw produktów: {e}")
            print("Będę używać tylko ID")
    
    def get_product_name(self, product_id):
        """Pobierz nazwę produktu po ID (zarówno oryginalnym jak i wewnętrznym)"""
        # Sprawdź czy to wewnętrzne ID (po mapowaniu)
        if product_id in self.id_to_product:
            original_id = self.id_to_product[product_id]
            if original_id in self.original_id_to_name:
                return self.original_id_to_name[original_id]
            else:
                return f"Product_{original_id}"
        
        # Sprawdź czy to oryginalne ID
        elif product_id in self.original_id_to_name:
            return self.original_id_to_name[product_id]
        
        # Jeśli nie ma mapowania
        else:
            return f"Product_{product_id}"
    
    def _map_and_replace(self, data_series, to_id_dict, id_to_dict, next_id):
        """
        Wewnętrzna, zoptymalizowana metoda do tworzenia mapowań i zamiany 
        oryginalnych wartości (ID produktu/kategorii) na wewnętrzne ID.
        """
        
        unique_values = data_series.explode().dropna().unique()
        
        # 2. Tworzenie nowych mapowań ID
        for value in unique_values:
            if value not in to_id_dict:
                to_id_dict[value] = next_id
                id_to_dict[next_id] = value
                next_id += 1
                
        # 3. Zastosowanie mapowania do serii wektorowo
        def map_list_of_ids(id_list, mapping_dict):
            return [mapping_dict.get(item) for item in id_list]

        mapped_series = data_series.apply(
            lambda x: map_list_of_ids(x, to_id_dict)
        )
        
        # Zwracamy zmapowaną serię i zaktualizowany następny ID
        return mapped_series, next_id
        
    def process_data(self, users_data):
        
        # Mapowanie i zamiana 'product_id'
        users_data['product_id'], self.next_product_id = self._map_and_replace(
            users_data['product_id'], 
            self.product_to_id, 
            self.id_to_product, 
            self.next_product_id
        )
        
        # Mapowanie i zamiana 'aisle_id'
        users_data['aisle_id'], self.next_category_id = self._map_and_replace(
            users_data['aisle_id'], 
            self.category_to_id, 
            self.id_to_category, 
            self.next_category_id
        )
        
        # NOWE: Zapisz mapowanie kategorii (aisle) do nazw
        self._load_aisle_names()
        
        return users_data
    
    def _load_aisle_names(self):
        """Załaduj nazwy alejek/kategorii"""
        try:
            aisles_file = Path(__file__).parent / "kaggle" / "archive" / "aisles.csv"
            aisles_df = pd.read_csv(aisles_file)
            
            self.aisle_id_to_name = {}
            for _, row in aisles_df.iterrows():
                self.aisle_id_to_name[row['aisle_id']] = row['aisle']
            
            print(f"✅ Załadowano nazwy {len(self.aisle_id_to_name):,} kategorii")
            
        except Exception as e:
            print(f"⚠️ Nie udało się załadować nazw kategorii: {e}")
            self.aisle_id_to_name = {}
    
    def get_category_name(self, category_id):
        """Pobierz nazwę kategorii po ID"""
        # Sprawdź czy to wewnętrzne ID
        if category_id in self.id_to_category:
            original_id = self.id_to_category[category_id]
            if original_id in self.aisle_id_to_name:
                return self.aisle_id_to_name[original_id]
        
        # Sprawdź czy to oryginalne ID
        elif category_id in self.aisle_id_to_name:
            return self.aisle_id_to_name[category_id]
        
        return f"Aisle_{category_id}"
    
    def get_vocab_size(self):
        return len(self.product_to_id)
    
    def get_num_categories(self):
        return len(self.category_to_id)


# GŁÓWNY KOD
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent  
    data_dir = BASE_DIR / "data"
    file_path = data_dir / "kaggle_prepared.csv"
    
    data = pd.read_csv(file_path, converters={
        'product_id': ast.literal_eval,
        'aisle_id': ast.literal_eval
    })
    
    print("=" * 60)
    print("ŁADOWANIE DANYCH I MAPOWANIE NAZW")
    print("=" * 60)
    
    productprocessor = ProductProcessor()
    user_data = productprocessor.process_data(data)
    user_data_df = pd.DataFrame(user_data) 
    
    print(f"\n✅ Przetworzono {len(user_data_df):,} wierszy")
    print(f"✅ Unikalnych produktów: {productprocessor.get_vocab_size():,}")
    print(f"✅ Unikalnych kategorii: {productprocessor.get_num_categories():,}")
    
    # Test: Pokaż przykładowe mapowanie
    print("\n🧪 TEST MAPOWANIA NAZW:")
    test_product_id = 196  # Z Twoich wyników
    test_category_id = 6   # Z Twoich wyników
    
    print(f"Produkt ID {test_product_id}: {productprocessor.get_product_name(test_product_id)}")
    print(f"Kategoria ID {test_category_id}: {productprocessor.get_category_name(test_category_id)}")
    
    print("\n" + "=" * 60)
    print("URUCHAMIANIE MODELU")
    print("=" * 60)
    
    trained_model = pytorchmodel.get_prediction(user_data_df, productprocessor)