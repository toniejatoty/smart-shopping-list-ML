from pathlib import Path
import pytorchmodel
from prepare_data_mapping_name_id import ProductProcessor

print("=" * 70)
print("TRAIN.PY - TRENING MODELU NA PRZETWORZONYCH DANYCH")
print("=" * 70)

# Ścieżki
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"
processed_dir = data_dir / "processed"

if not (processed_dir / 'processed_data.pkl').exists():
    print("\n❌ BŁĄD: Brak przetworzonych danych!")
    print(f"   Uruchom najpierw: python prepare_data_mapping_name_id.py")
    print(f"   Spodziewany plik: {processed_dir / 'processed_data.pkl'}")
    exit(1)

print("\n📥 WCZYTYWANIE MAPPINGS...")
productprocessor = ProductProcessor()
user_data_df = productprocessor.load_processed_data(processed_dir)

print(f"\n📊 STATYSTYKI WCZYTANYCH DANYCH:")
print(f"  Wierszy: {len(user_data_df):,}")
print(f"  Użytkowników: {user_data_df['user_id'].nunique():,}")
print(f"  Unikalnych produktów: {productprocessor.get_vocab_size():,}")
print(f"  Unikalnych kategorii: {productprocessor.get_num_categories():,}")
print(f"  Max produktów w koszyku: {productprocessor.max_basket_len}")
print(f"  Max kategorii per produkt: {productprocessor.max_cats_per_product}")

# 2. Uruchom trening
print("\n" + "=" * 70)
print("URUCHAMIANIE TRENINGU MODELU")
print("=" * 70)

trained_model = pytorchmodel.get_prediction(user_data_df, productprocessor)

print("\n" + "=" * 70)
print("✅ TRENING ZAKOŃCZONY POMYŚLNIE!")
print("=" * 70)


