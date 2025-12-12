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
        self.next_off_product_id = 1
        self.next_category_id = 1
        
        # NOWE: Mapowanie oryginalnych ID do nazw produktów
        self.original_id_to_name = {}
        self._load_product_names()
    
    def _load_product_names(self):
        """Załaduj mapowanie ID produktu -> nazwa z pliku products.csv"""
        try:
            project_root = Path(__file__).parent.parent
            products_file = project_root / 'prepare_data_for_model' / 'OpenFoodData.csv'
            products_df = pd.read_csv(products_file)
            
            # Stwórz mapowanie: off_product_id -> product_name
            for _, row in products_df.iterrows():
                self.original_id_to_name[row['Id']] = row['Name']
            
            print(f"✅ Załadowano nazwy {len(self.original_id_to_name):,} produktów")
            
            # Sprawdź przykładowe mapowanie
            sample_ids = list(self.original_id_to_name.keys())[:5]
            print("Przykładowe mapowania:")
            for pid in sample_ids:
                print(f"  {pid} -> {self.original_id_to_name[pid][:30]}...")
                
        except Exception as e:
            print(f"⚠️ Nie udało się załadować nazw produktów: {e}")
            print("Będę używać tylko ID")
    
    def get_product_name(self, off_product_id):
        """Pobierz nazwę produktu po ID (zarówno oryginalnym jak i wewnętrznym)"""
        # Sprawdź czy to wewnętrzne ID (po mapowaniu)
        if off_product_id in self.id_to_product:
            original_id = self.id_to_product[off_product_id]
            if original_id in self.original_id_to_name:
                return self.original_id_to_name[original_id]
            else:
                return f"Product_{original_id}"
        
        # Sprawdź czy to oryginalne ID
        elif off_product_id in self.original_id_to_name:
            return self.original_id_to_name[off_product_id]
        
        # Jeśli nie ma mapowania
        else:
            return f"Product_{off_product_id}"
    
    def _map_and_replace(self, data_series, to_id_dict, id_to_dict, next_id, is_product=True):
        """
        Zmieniona wersja: rozróżnia produkty i kategorie
        is_product=True: dla produktów (płaska lista)
        is_product=False: dla kategorii (lista list)
        """
        
        # 1. Zbierz wszystkie unikalne wartości
        all_values = set()
        
        for row in data_series:
            if isinstance(row, list):
                for item in row:
                    if isinstance(item, list) and not is_product:
                        # Dla kategorii: zbierz wszystkie elementy z listy list
                        for subitem in item:
                            if subitem is not None and str(subitem) != 'nan':
                                all_values.add(str(subitem))
                    elif item is not None and str(item) != 'nan':
                        all_values.add(str(item))
        
        # 2. Tworzenie nowych mapowań
        for value in all_values:
            if value not in to_id_dict:
                to_id_dict[value] = next_id
                id_to_dict[next_id] = value
                next_id += 1
        
        # 3. RÓŻNE MAPOWANIE DLA PRODUKTÓW I KATEGORII
        if is_product:
            # DLA PRODUKTÓW: płaska lista ID
            def map_products(product_list, mapping_dict):
                """Mapuj listę produktów na listę ID"""
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
        
        # Mapowanie i zamiana 'off_product_id'
        users_data['off_product_id'], self.next_off_product_id = self._map_and_replace(
            users_data['off_product_id'], 
            self.product_to_id, 
            self.id_to_product, 
            self.next_off_product_id,
            is_product=True
        )
        
        # Mapowanie i zamiana 'aisle_id'
        users_data['aisle_id'], self.next_category_id = self._map_and_replace(
            users_data['aisle_id'], 
            self.category_to_id, 
            self.id_to_category, 
            self.next_category_id,
            is_product=False
        )
        
        # NOWE: Zapisz mapowanie kategorii (aisle) do nazw
        self._load_aisle_names()
        
        return users_data
    
    def _load_aisle_names(self):
        """Załaduj nazwy alejek/kategorii"""
        try:
            project_root = Path(__file__).parent.parent
            aisles_file = project_root / 'prepare_data_for_model' / 'kaggle' / 'archive' / 'aisles.csv'
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
    
    def safe_literal_eval(x):
        """Bezpieczna konwersja stringa na listę"""
        if pd.isna(x):
            return []
        
        x_str = str(x).strip()
        
        # Jeśli to puste
        if not x_str or x_str == '[]' or x_str == 'nan':
            return []
        
        # Sprawdź czy zaczyna się od [
        if x_str.startswith('[') and x_str.endswith(']'):
            try:
                return ast.literal_eval(x_str)
            except (SyntaxError, ValueError):
                # Jeśli nie uda się sparsować, spróbuj ręcznie
                x_str = x_str[1:-1]  # usuń nawiasy
                if ',' in x_str:
                    items = [item.strip().strip("'\"") for item in x_str.split(',')]
                    return items
                elif x_str:
                    return [x_str]
                else:
                    return []
        else:
            # Zwykły string
            return [x_str]

    data = pd.read_csv(file_path, converters={
        'off_product_id': safe_literal_eval,
        'aisle_id': safe_literal_eval
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
    test_off_product_id = 196  # Z Twoich wyników
    test_category_id = 6   # Z Twoich wyników
    
    print(f"Produkt ID {test_off_product_id}: {productprocessor.get_product_name(test_off_product_id)}")
    print(f"Kategoria ID {test_category_id}: {productprocessor.get_category_name(test_category_id)}")
    
    print("\n" + "=" * 60)
    print("URUCHAMIANIE MODELU")
    print("=" * 60)
    
    trained_model = pytorchmodel.get_prediction(user_data_df, productprocessor)