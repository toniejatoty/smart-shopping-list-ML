# simple_product_mapping.py
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURACJA ==========
BASE_DIR = Path(__file__).parent
KAGGLE_DIR = BASE_DIR / "kaggle" / "archive"
OFF_FILE = BASE_DIR / "OpenFoodData.csv"
MODEL_DIR = BASE_DIR.parent.parent / "product_category_model"

# ========== 1. WCZYTAJ DANE ==========
print("=" * 70)
print("1. WCZYTYWANIE DANYCH")
print("=" * 70)

#Produkty Kaggle
products = pd.read_csv(KAGGLE_DIR / "products.csv")
print(f"✅ Kaggle products: {len(products)}")

# Wczytaj OFF
off_df = pd.read_csv(OFF_FILE)
print(f"✅ OFF products: {len(off_df)}")

# ========== 2. WCZYTAJ MODEL ==========
print("\n" + "=" * 70)
print("2. ŁADOWANIE MODELU")
print("=" * 70)

model = SentenceTransformer(str(MODEL_DIR))
print(f"✅ Model loaded: {model.max_seq_length}")

# ========== 3. MAPOWANIE PRODUKTÓW ==========
print("\n" + "=" * 70)
print("3. MAPOWANIE PRODUKTÓW KAGGLE → OFF")
print("=" * 70)

def simple_product_mapper(kaggle_products, off_df, model):
    """Proste mapowanie produktów - tylko ID"""
    
    # Przygotuj OFF produkty
    off_products = []
    for _, row in off_df.iterrows():
        off_id = str(row['Id']) if 'Id' in row and pd.notna(row['Id']) else None
        name = str(row['Name']) if 'Name' in row and pd.notna(row['Name']) else ''
        
        if off_id and name:
            off_products.append({
                'id': off_id,
                'name': name.lower().strip(),
                'original_name': name
            })
    
    print(f"OFF products: {len(off_products)}")
    
    # Get embeddings dla OFF
    print("Creating OFF embeddings...")
    off_names = [p['original_name'] for p in off_products]
    off_embeddings = model.encode(off_names, show_progress_bar=True)
    off_embeddings_norm = off_embeddings / np.linalg.norm(off_embeddings, axis=1, keepdims=True) # normalization becouse long product names have larger embeddings, norm 2 after this is 1
    
    # Mapuj Kaggle produkty
    product_mapping = {}
    batch_size = 1000
    
    for i in range(0, len(kaggle_products), batch_size):
        batch = kaggle_products.iloc[i:i+batch_size]
        print(f"Batch {i//batch_size + 1}/{(len(kaggle_products)-1)//batch_size + 1}...")
        
        kaggle_names = batch['product_name'].astype(str).tolist()
        kaggle_embeddings = model.encode(kaggle_names, show_progress_bar=False)
        kaggle_embeddings_norm = kaggle_embeddings / np.linalg.norm(kaggle_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(kaggle_embeddings_norm, off_embeddings_norm.T)
        
        for idx, (_, row) in enumerate(batch.iterrows()):
            best_idx = np.argmax(similarities[idx])
            best_score = similarities[idx, best_idx]
            
            if best_score > 0.6:
                off_product = off_products[best_idx]
                product_mapping[row['product_id']] = off_product['id']
            else:
                # Text fallback
                k_name = str(row['product_name']).lower().strip()
                best_match = None
                
                for o_product in off_products:
                    o_name = o_product['name']
                    
                    if k_name in o_name or o_name in k_name:
                        best_match = o_product
                        break
                    else:
                        k_tokens = set(k_name.split())
                        o_tokens = set(o_name.split())
                        if k_tokens and o_tokens:
                            common = k_tokens.intersection(o_tokens)
                            if common and len(common) / max(len(k_tokens), len(o_tokens)) > 0.5:
                                best_match = o_product
                                break
                
                if best_match:
                    product_mapping[row['product_id']] = best_match['id']
                else:
                    # No match - użyj specjalnej wartości
                    product_mapping[row['product_id']] = 'NO_MATCH'
    
    return product_mapping

# Uruchom mapowanie
print("Rozpoczynam mapowanie...")
mapping_dict = simple_product_mapper(products, off_df, model)

# ========== 4. ZAPISZ MAPOWANIE ==========
print("\n" + "=" * 70)
print("4. ZAPISZ MAPOWANIE DO CSV")
print("=" * 70)

# Stwórz prosty DataFrame
mapping_list = []
for kaggle_id, off_id in mapping_dict.items():
    mapping_list.append({
        'kaggle_product_id': kaggle_id,
        'off_product_id': off_id  # 'NO_MATCH' jeśli brak mapowania
    })

mapping_df = pd.DataFrame(mapping_list)

# Zapisz do CSV
output_file = BASE_DIR / 'simple_product_mapping.csv'
mapping_df.to_csv(output_file, index=False)

print(f"✅ Utworzono plik: {output_file}")
print(f"   Zmapowano {len(mapping_df)} produktów")

# Statystyki
mapped = mapping_df[mapping_df['off_product_id'] != 'NO_MATCH']
unmapped = mapping_df[mapping_df['off_product_id'] == 'NO_MATCH']

print(f"\n📊 STATYSTYKI:")
print(f"   Wszystkich produktów: {len(mapping_df)}")
print(f"   Zmapowanych: {len(mapped)} ({len(mapped)/len(mapping_df)*100:.1f}%)")
print(f"   Bez mapowania: {len(unmapped)} ({len(unmapped)/len(mapping_df)*100:.1f}%)")

# Przykładowe dane
print(f"\n📋 PRZYKŁADOWE DANE:")
print("Pierwsze 10 wierszy:")
print(mapping_df.head(10).to_string())

print(f"\n🎉 GOTOWE! Masz plik z mapowaniem ID.")