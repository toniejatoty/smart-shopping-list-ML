# Autor: ekspert

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ShoppingDataset
from model import TransformerShoppingListRecommender
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEQ_LEN = 8
MAX_BASKET = 50
MAX_CAT = 20
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 128
TOP_K = 10
TEMPERATURE = 0.1
NUM_PRODUCTS = 6743
NUM_CATEGORIES = 1731

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("WCZYTYWANIE DANYCH")
print("="*60)

project_root = Path(__file__).parent.parent
csv_path = project_root / "ml" / "data" / "model_input.csv"

print(f"Ścieżka do danych: {csv_path}")
print(f"Istnieje: {csv_path.exists()}")

df = pd.read_csv(csv_path)
print(f"Wczytano {len(df)} wierszy")

print("Parsowanie JSON (produkty)...")
tqdm.pandas(desc="Produkty")
df['off_product_id'] = df['off_product_id'].progress_apply(json.loads)

print("Parsowanie JSON (kategorie)...")
tqdm.pandas(desc="Kategorie")
df['mapped_categories'] = df['mapped_categories'].progress_apply(json.loads)

print("\nGrupowanie użytkowników...")
user_groups = [g.reset_index(drop=True) for _, g in tqdm(df.groupby('user_id'), desc="Grupowanie") if len(g) > 1]
print(f"Liczba użytkowników z >1 zamówieniem: {len(user_groups)}")

print("\nPodział train/val...")
_, val_users = train_test_split(user_groups, test_size=0.2, random_state=42)
print(f"Użytkownicy walidacyjni: {len(val_users)}")

print("\nTworzenie datasetu walidacyjnego...")
val_dataset = ShoppingDataset(val_users, seq_len=SEQ_LEN, max_basket_size=MAX_BASKET,
                              max_cat_size=MAX_CAT, max_samples_per_user=5)
print(f"Liczba sampli walidacyjnych: {len(val_dataset)}")

print(f"Tworzenie DataLoader (batch_size={BATCH_SIZE})...")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
print(f"Liczba batchy: {len(val_loader)}")

print("\n" + "="*60)
print("ŁADOWANIE MODELU")
print("="*60)

model_path = project_root / "recommender_model_v3.pth"
print(f"Ścieżka modelu: {model_path}")
print(f"Istnieje: {model_path.exists()}")
print(f"Urządzenie: {device}")

print("\nTworzenie architektury modelu...")
model = TransformerShoppingListRecommender(
    num_products=NUM_PRODUCTS,
    num_categories=NUM_CATEGORIES,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    sequence_length=SEQ_LEN,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametry całkowite: {total_params:,}")
print(f"Parametry trenowalne: {trainable_params:,}")

print("\nŁadowanie wag modelu...")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print("✓ Model załadowany pomyślnie")

print("\n" + "="*60)
print("EWALUACJA")
print("="*60)

total_recall = 0.0
total_precision = 0.0
total_hits = 0
total_true_items = 0
num_samples = 0

print("\nPre-computing embeddings produktów...")
with torch.no_grad():
    all_products = torch.arange(NUM_PRODUCTS, device=device).unsqueeze(1)
    all_cats = torch.zeros((NUM_PRODUCTS, 1, MAX_CAT), dtype=torch.long, device=device)
    prod_vecs = model.encode_product(all_products, all_cats)
    prod_norm = F.normalize(prod_vecs, p=2, dim=1)
    print(f"✓ Zakodowano {NUM_PRODUCTS} produktów (shape: {prod_norm.shape})")

print(f"\nRozpoczynanie ewaluacji na {len(val_loader)} batchach...\n")

with torch.no_grad():
    pbar = tqdm(val_loader, desc="Ewaluacja", unit="batch")
    for batch_idx, (in_prod, in_cat, in_time, target_prod, target_cat) in enumerate(pbar):
        in_prod, in_cat, in_time = in_prod.to(device), in_cat.to(device), in_time.to(device)
        target_prod = target_prod.to(device)

        pred_vec = model(in_prod, in_cat, in_time)
        pred_norm = F.normalize(pred_vec, p=2, dim=1)

        sims = torch.matmul(pred_norm, prod_norm.T)
        topk_indices = torch.topk(sims, k=TOP_K, dim=1).indices

        batch_recall = 0.0
        batch_precision = 0.0
        batch_hits = 0
        
        for i in range(target_prod.size(0)):
            true_items = target_prod[i][target_prod[i] != 0].tolist()
            pred_items = topk_indices[i].tolist()
            
            hits = len(set(true_items) & set(pred_items))
            recall = hits / max(len(true_items), 1)
            precision = hits / TOP_K
            
            total_recall += recall
            total_precision += precision
            total_hits += hits
            total_true_items += len(true_items)
            num_samples += 1
            
            batch_recall += recall
            batch_precision += precision
            batch_hits += hits
        
        avg_batch_recall = batch_recall / target_prod.size(0)
        avg_batch_precision = batch_precision / target_prod.size(0)
        pbar.set_postfix({
            'R': f'{avg_batch_recall:.3f}',
            'P': f'{avg_batch_precision:.3f}',
            'hits': batch_hits
        })

avg_recall = total_recall / num_samples
avg_precision = total_precision / num_samples
avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
avg_hits_per_sample = total_hits / num_samples

print("\n" + "="*60)
print("WYNIKI KOŃCOWE")
print("="*60)
print(f"Model: {model_path.name}")
print(f"Top-K: {TOP_K}")
print(f"Liczba sampli: {num_samples}")
print(f"Średnia długość koszyka: {total_true_items/num_samples:.2f}")
print("\nMetryki:")
print(f"  Recall@{TOP_K}:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
print(f"  Precision@{TOP_K}: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
print(f"  F1-Score@{TOP_K}:  {avg_f1:.4f} ({avg_f1*100:.2f}%)")
print(f"  Avg Hits:       {avg_hits_per_sample:.2f}")
print("="*60)
