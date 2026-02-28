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
TOP_K = 10
NUM_NEG = 100
POOL_SIZE = 1000        
SEMI_HARD_MARGIN = 0.3  
SUBSET_FRACTION = 1
EARLY_STOPPING_PATIENCE = 3


def mine_negatives(pred_norm, pos_norm, num_products, num_neg, model, device, epoch, total_epochs, product_cats_tensor):
    batch_size = pred_norm.size(0)

    progress = epoch / max(total_epochs - 1, 1)
    if progress < 0.3:
        hard_ratio = 0.0
    elif progress < 0.6:
        hard_ratio = 0.5
    else:
        hard_ratio = 0.8

    num_hard = int(num_neg * hard_ratio)
    num_random_neg = num_neg - num_hard

    random_idx = torch.randint(0, num_products, (batch_size, num_random_neg), device=device)

    if num_hard > 0:
        pool_idx = torch.randint(0, num_products, (POOL_SIZE,), device=device)
        with torch.no_grad():
            pool_prods = pool_idx.unsqueeze(1)
            pool_cats = product_cats_tensor[pool_idx].unsqueeze(1)  
            pool_vecs = model.encode_product(pool_prods, pool_cats)
            pool_norm = F.normalize(pool_vecs, p=2, dim=1)

        pos_sim_vals = (pred_norm * pos_norm).sum(dim=1)          
        pool_sim = torch.mm(pred_norm.detach(), pool_norm.T)       

        ps = pos_sim_vals.unsqueeze(1)                             
        semi_hard_mask = (pool_sim < ps) & (pool_sim > ps - SEMI_HARD_MARGIN)

        masked_sims = pool_sim.clone()
        masked_sims[~semi_hard_mask] = -2.0
        top_indices = masked_sims.topk(num_hard, dim=1).indices    
        hard_idx = pool_idx[top_indices]                           

        valid_counts = semi_hard_mask.sum(dim=1)                   
        fallback_mask = valid_counts < num_hard
        if fallback_mask.any():
            fallback_indices = pool_sim.topk(num_hard, dim=1).indices
            fallback_hard = pool_idx[fallback_indices]
            hard_idx[fallback_mask] = fallback_hard[fallback_mask]

        all_idx = torch.cat([random_idx, hard_idx], dim=1)
    else:
        all_idx = random_idx

    flat_prods = all_idx.view(-1, 1)                                           
    flat_ids = flat_prods.squeeze(1)                                           
    flat_cats = product_cats_tensor[flat_ids].unsqueeze(1)                     
    all_vecs = model.encode_product(flat_prods, flat_cats)
    return all_vecs.view(batch_size, num_neg, -1)


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

    print("Budowanie mappingu produkt → kategorie...")
    product_cats = [[]] * num_products
    for prods, cat_lists in zip(final_data['off_product_id'], final_data['mapped_categories']):
        for prod_id, cats in zip(prods, cat_lists):
            if product_cats[prod_id] == []:
                product_cats[prod_id] = cats
    product_cats_np = torch.zeros(num_products, MAX_CAT, dtype=torch.long)
    for prod_id, cats in enumerate(product_cats):
        cats_trimmed = cats[:MAX_CAT]
        product_cats_np[prod_id, :len(cats_trimmed)] = torch.tensor(cats_trimmed, dtype=torch.long)
    print(f"Mapping gotowy: {product_cats_np.shape}")

    print("Grupowanie użytkowników...")
    user_groups = [group.reset_index(drop=True) for _, group in final_data.groupby('user_id') if len(group) > 1]

    print("Podział na Train/Val (według użytkowników)...")
    if SUBSET_FRACTION < 1.0:
        import random
        random.seed(42)
        user_groups = random.sample(user_groups, int(len(user_groups) * SUBSET_FRACTION))
        print(f"Używam {SUBSET_FRACTION*100:.0f}% użytkowników ({len(user_groups)})")
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
    product_cats_tensor = product_cats_np.to(device)

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

    best_val_recall = 0.0
    epochs_no_improve = 0

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
            neg_vecs = mine_negatives(
                pred_norm, pos_norm, num_products, NUM_NEG,
                model, device, epoch, EPOCHS, product_cats_tensor
            )
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

        total_recall = 0.0
        num_samples = 0
        with torch.no_grad():
            all_products = torch.arange(num_products, device=device).unsqueeze(1)
            all_cats = product_cats_tensor.unsqueeze(1)
            prod_vecs = model.encode_product(all_products, all_cats)
            prod_norm = F.normalize(prod_vecs, p=2, dim=1)

            for in_prod, in_cat, in_time, target_prod, target_cat in val_loader:
                in_prod, in_cat, in_time = in_prod.to(device), in_cat.to(device), in_time.to(device)
                target_prod = target_prod.to(device)

                pred_vec = model(in_prod, in_cat, in_time)
                pred_norm_r = F.normalize(pred_vec, p=2, dim=1)
                sims = torch.matmul(pred_norm_r, prod_norm.T)
                topk_indices = torch.topk(sims, k=TOP_K, dim=1).indices

                for i in range(target_prod.size(0)):
                    true_items = target_prod[i][target_prod[i] != 0].tolist()
                    pred_items = topk_indices[i].tolist()
                    hits = len(set(true_items) & set(pred_items))
                    total_recall += hits / max(len(true_items), 1)
                    num_samples += 1

        avg_val_recall = total_recall / max(num_samples, 1)

        print(f"\n--- PODSUMOWANIE EPOKI {epoch+1} ---")
        print(f"Train Loss:    {avg_train_loss:.4f}")
        print(f"Val Loss:      {avg_val_loss:.4f}")
        print(f"Recall@{TOP_K}:    {avg_val_recall:.4f} ({avg_val_recall*100:.2f}%)")

        if avg_val_recall > best_val_recall:
            best_val_recall = avg_val_recall
            epochs_no_improve = 0
            model_path = project_root / "recommender_model_v3.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model zapisany (Recall@{TOP_K}={avg_val_recall:.4f}): {model_path}")
        else:
            epochs_no_improve += 1
            print(f"Brak poprawy: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping po epoce {epoch+1}. Najlepszy Recall@{TOP_K}: {best_val_recall:.4f}")
                break

if __name__ == '__main__':
    main()