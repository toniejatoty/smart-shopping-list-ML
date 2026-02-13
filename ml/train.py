import gc
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from model import TransformerShoppingListRecommender
from sklearn.model_selection import train_test_split
from dataset import ShoppingDataset
import pandas as pd
import json
from tqdm import tqdm

SEQ_LEN = 8
MAX_BASKET = 50
MAX_CAT = 20
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 128
TEMPERATURE = 0.1
EPOCHS = 10
NUM_NEG = 100

def main():
    tqdm.pandas()

    print("Wczytywanie danych...")
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "ml" / "data" / "model_input.csv"
    final_data = pd.read_csv(csv_path)

    print("Parsowanie list (JSON)...")
    final_data['off_product_id'] = final_data['off_product_id'].progress_apply(json.loads)
    final_data['mapped_categories'] = final_data['mapped_categories'].progress_apply(json.loads)

    print("Obliczanie zakresów ID produktów...")
    max_p = 0
    for prods in tqdm(final_data['off_product_id'], desc="Produkty"):
        if prods:
            m = max(prods)
            if m > max_p: max_p = m
    num_products = int(max_p + 1)

    print("Obliczanie zakresów ID kategorii...")
    max_c = 0
    for cat_lists in tqdm(final_data['mapped_categories'], desc="Kategorie"):
        for clist in cat_lists:
            if clist:
                m = max(clist)
                if m > max_c: max_c = m
    num_categories = int(max_c + 1)

    print(f"Liczba produktów: {num_products}, Liczba kategorii: {num_categories}")

    print("Grupowanie użytkowników...")
    user_groups = [group.reset_index(drop=True) for _, group in final_data.groupby('user_id') if len(group) > 1]

    print("Podział na Train/Val (według użytkowników)...")
    train_users, val_users = train_test_split(user_groups, test_size=0.2, random_state=42)

    train_dataset = ShoppingDataset(train_users, seq_len=SEQ_LEN, max_basket_size=MAX_BASKET, max_cat_size=MAX_CAT, max_samples_per_user=5)
    val_dataset = ShoppingDataset(val_users, seq_len=SEQ_LEN, max_basket_size=MAX_BASKET, max_cat_size=MAX_CAT, max_samples_per_user=5)

    print(f"Liczba sampli treningowych: {len(train_dataset)}")
    print(f"Liczba sampli walidacyjnych: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    del final_data
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Tworzenie modelu...")
    model = TransformerShoppingListRecommender(
        num_products=num_products,
        num_categories=num_categories,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQ_LEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0004,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS} - TRAIN", unit="batch")

        for in_prod, in_cat, in_time, target_prod, target_cat in train_pbar:
            in_prod, in_cat, in_time = in_prod.to(device), in_cat.to(device), in_time.to(device)
            target_prod, target_cat = target_prod.to(device), target_cat.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred_vector = model(in_prod, in_cat, in_time)
            pos_vector = model.encode_product(target_prod, target_cat)

            pred_norm = F.normalize(pred_vector, p=2, dim=1)
            pos_norm = F.normalize(pos_vector, p=2, dim=1)

            batch_size = pred_norm.size(0)
            neg_indices = torch.randint(0, num_products, (batch_size, NUM_NEG), device=device)
            for b in range(batch_size):
                target_set = set(target_prod[b].tolist())
                mask = torch.tensor([i not in target_set for i in neg_indices[b]], device=device)
                neg_indices[b] = neg_indices[b][mask]
                while neg_indices[b].size(0) < NUM_NEG:
                    neg_indices[b] = torch.cat([neg_indices[b], torch.randint(0, num_products, (1,), device=device)], dim=0)

            flat_neg_prods = neg_indices.view(-1,1)
            flat_neg_cats = torch.zeros((batch_size * NUM_NEG,1, MAX_CAT), dtype=torch.long, device=device)
            flat_neg_vecs = model.encode_product(flat_neg_prods, flat_neg_cats)
            neg_vecs = flat_neg_vecs.view(batch_size, NUM_NEG, -1)
            neg_norm = F.normalize(neg_vecs, p=2, dim=2)

            pos_sim = torch.sum(pred_norm * pos_norm, dim=1, keepdim=True)
            neg_sim = torch.einsum('bd,bkd->bk', pred_norm, neg_norm)

            logits = torch.cat([pos_sim, neg_sim], dim=1) / TEMPERATURE
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            del pred_vector, pos_vector, pred_norm, pos_norm, neg_vecs, neg_norm, logits, loss

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for in_prod, in_cat, in_time, target_prod, target_cat in val_loader:
                in_prod, in_cat, in_time = in_prod.to(device), in_cat.to(device), in_time.to(device)
                target_prod, target_cat = target_prod.to(device), target_cat.to(device)

                pred_vector = model(in_prod, in_cat, in_time)
                pos_vector = model.encode_product(target_prod, target_cat)

                pred_norm = F.normalize(pred_vector, p=2, dim=1)
                pos_norm = F.normalize(pos_vector, p=2, dim=1)

                batch_size = pred_norm.size(0)
                neg_indices = torch.randint(0, num_products, (batch_size, NUM_NEG), device=device)
                flat_neg_prods = neg_indices.view(-1,1) 
                flat_neg_cats = torch.zeros((batch_size * NUM_NEG,1, MAX_CAT), dtype=torch.long, device=device)
                flat_neg_vecs = model.encode_product(flat_neg_prods, flat_neg_cats)
                neg_vecs = flat_neg_vecs.view(batch_size, NUM_NEG, -1)
                neg_norm = F.normalize(neg_vecs, p=2, dim=2)

                pos_sim = torch.sum(pred_norm * pos_norm, dim=1, keepdim=True)
                neg_sim = torch.einsum('bd,bkd->bk', pred_norm, neg_norm)
                logits = torch.cat([pos_sim, neg_sim], dim=1) / TEMPERATURE
                labels = torch.zeros(batch_size, dtype=torch.long, device=device)

                loss = F.cross_entropy(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\n--- PODSUMOWANIE EPOKI {epoch+1} ---")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")

        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = project_root / "recommender_model_v3.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model zapisany: {model_path}")

if __name__ == '__main__':
    main()
