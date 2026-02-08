# pytorchmodel_with_categories.py
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import datetime
from pathlib import Path


class LSTMRecommenderWithCategoryMean(nn.Module):
    def __init__(self, num_products, num_categories, 
                 category_embedding_dim, 
                 product_embedding_dim, 
                 timestamp_embedding_dim,
                 hidden_dim, 
                 sequence_length, 
                 num_layers, 
                 dropout):
        super().__init__()
        self.num_products = num_products
        self.num_categories = num_categories
        self.sequence_length = sequence_length
        
        # Embeddingi
        self.product_embedding = nn.Embedding(num_products + 1, product_embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, category_embedding_dim, padding_idx=0)
        
        # Timestamp
        self.timestamp_embedding = nn.Embedding(31, timestamp_embedding_dim, padding_idx=None)
        
        # Wymiar wejściowy dla LSTM
        input_dim = product_embedding_dim + category_embedding_dim + timestamp_embedding_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_products + 1)
    
    def forward(self, products_tensor, categories_tensor, timestamps_tensor, batch_size):
        """
        products_tensor: [batch_size * seq_length, max_basket_len]
        categories_tensor: [batch_size * seq_length, max_basket_len, max_categories]
        timestamps_tensor: [batch_size, seq_length]
        """
        
        # === CZĘŚĆ 1: PRODUKTY ===
        # 2. Embedding produktów
        product_embeds = self.product_embedding(products_tensor)  # [batch_size_seq, max_basket_len, product_embed_dim]
        
        # 3. Maska produktów (które są prawdziwe, nie padding)
        product_mask = (products_tensor != 0).float().unsqueeze(-1)  # [batch_size_seq, max_basket_len, 1]
        
        # 4. Średnia embeddingów produktów w koszyku
        sum_product_embeds = torch.sum(product_embeds * product_mask, dim=1)  # [batch_size_seq, product_embed_dim]
        num_products_in_basket = torch.clamp(torch.sum(product_mask, dim=1), min=1.0)  # [batch_size_seq, 1]
        product_mean = sum_product_embeds / num_products_in_basket  # [batch_size_seq, product_embed_dim]
        
        # === CZĘŚĆ 2: KATEGORIE ===
        # 5. Pobierz rozmiary kategorii
        batch_size_seq_cat, basket_size, max_cats = categories_tensor.shape
        
        # 6. Embedding kategorii dla WSZYSTKICH kategorii
        flat_categories = categories_tensor.view(-1)  # [batch_size_seq * basket_size * max_cats]
        cat_embeddings_all = self.category_embedding(flat_categories)  # [N, category_embed_dim]
        
        # 7. Reshape z powrotem do struktury: produkty × kategorie
        cat_embeddings = cat_embeddings_all.view(
            batch_size_seq_cat, basket_size, max_cats, -1
        )  # [batch_size_seq, basket_size, max_cats, category_embed_dim]
        
        # 8. Maska kategorii (które są prawdziwe, nie padding)
        cat_mask = (categories_tensor != 0).float().unsqueeze(-1)  # [batch_size_seq, basket_size, max_cats, 1]
        
        # 9. Średnia kategorii DLA KAŻDEGO PRODUKTU
        sum_cat_per_product = torch.sum(cat_embeddings * cat_mask, dim=2)  # [batch_size_seq, basket_size, category_embed_dim]
        num_cats_per_product = torch.clamp(torch.sum(cat_mask, dim=2), min=1.0)  # [batch_size_seq, basket_size, 1]
        mean_cat_per_product = sum_cat_per_product / num_cats_per_product  # [batch_size_seq, basket_size, category_embed_dim]
        
        # 10. Średnia mean kategorii w CAŁYM KOSZYKU
        # Broadcast: [batch_size_seq, basket_size, category_embed_dim] * [batch_size_seq, basket_size, 1]
        masked_cat_means = mean_cat_per_product * product_mask  # [batch_size_seq, basket_size, category_embed_dim]
        sum_cat_embeds = torch.sum(masked_cat_means, dim=1)  # [batch_size_seq, category_embed_dim]
        category_mean = sum_cat_embeds / num_products_in_basket  # [batch_size_seq, category_embed_dim]
        
        # === CZĘŚĆ 3: TIMESTAMP ===
        # 11. Embedding timestampów
        timestamps_flat = timestamps_tensor.view(-1)  # [batch_size * seq_length]
        timestamp_embed = self.timestamp_embedding(timestamps_flat)  # [batch_size * seq_length, timestamp_embed_dim]
        
        # === CZĘŚĆ 4: POŁĄCZENIE ===
        # 12. Połącz wszystkie embeddingi
        combined_embed = torch.cat([product_mean, category_mean, timestamp_embed], dim=1)
        # [batch_size_seq, product_embed_dim + category_embed_dim + timestamp_embed_dim]
        
        # === CZĘŚĆ 5: LSTM ===
        # 13. Reshape dla LSTM: [batch, seq, features]
        lstm_input = combined_embed.view(batch_size, self.sequence_length, -1)
        # [batch_size, seq_length, combined_features]
        
        # 14. Przetwarzanie LSTM
        lstm_out, _ = self.lstm(lstm_input)
        last_output = lstm_out[:, -1, :]  # Bierzemy ostatni output sekwencji
        # [batch_size, hidden_dim]
        
        # === CZĘŚĆ 6: WARSTWY GĘSTE ===
        # 15. Warstwy fully connected
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        # [batch_size, num_products + 1]
        
        return logits

class ProductSequenceDatasetWithCategories(Dataset):
    def __init__(self, user_histories, product_processor, sequence_length):
        self.user_histories = user_histories
        self.product_processor = product_processor
        self.sequence_length = sequence_length
        self.num_products = product_processor.get_vocab_size()
        
        
        self.input_basket_products_sequences = []  
        self.input_basket_categories_sequences = []
        self.target_baskets = []          
        self.user_timestamps = []         
        
        print(f"\n=== INFORMACJE O DATASET Z KATEGORIAMI ===")
        print(f"Kolumny w danych: {list(user_histories.columns)}")
        print(f"Liczba wierszy: {len(user_histories):,}")
        print(f"Unikalnych produktów: {self.num_products}")
        print(f"Unikalnych nazw kategorii: {self.product_processor.get_num_categories()}")
        
        self._prepare_sequences()
    
    
    def _prepare_sequences(self):
        print("\nPrzygotowywanie sekwencji z listami list kategorii...")
        
        grouped_user_data = self.user_histories.groupby('user_id')
        total_users = self.user_histories['user_id'].nunique()
        
        print(f"Przetwarzanie {total_users:,} unikalnych użytkowników...")
        
        sequences_count = 0
        for user_id, user_data in tqdm(grouped_user_data, desc="Użytkownicy", total=total_users):
            
            if len(user_data) > self.sequence_length:
                for seq_start in range(len(user_data) - self.sequence_length):
                    # Input products
                    input_products = list(user_data['off_product_id'].iloc[seq_start:seq_start + self.sequence_length])
                    
                    raw_categories = list(user_data['aisle_id'].iloc[seq_start:seq_start + self.sequence_length])
                    target_basket = list(user_data['off_product_id'].iloc[seq_start + self.sequence_length])
                    timestamps = list(user_data['days_since_prior_order'].iloc[seq_start:seq_start + self.sequence_length])
                    
                    self.input_basket_products_sequences.append(input_products)
                    self.input_basket_categories_sequences.append(raw_categories)
                    self.target_baskets.append(target_basket)
                    self.user_timestamps.append(timestamps)
                    
                    sequences_count += 1
        
        print(f"\nUtworzono {sequences_count:,} sekwencji")
    
    def __len__(self):
        return len(self.input_basket_products_sequences)
    
    def __getitem__(self, idx):
        target_basket = self.target_baskets[idx]
        target_vector = torch.zeros(self.num_products + 1, dtype=torch.float)
        
        for off_product_id in target_basket:
            target_vector[off_product_id] = 1.0
        
        return (
            self.input_basket_products_sequences[idx],  
            self.input_basket_categories_sequences[idx],
            self.user_timestamps[idx],
            target_vector
        )

def train_model_with_categories(model, train_dataloader, 
                               num_epochs, learning_rate, K):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"\nTrening na: {device}")
    
    # Oblicz pos_weight
    total_positive_items = 0
    total_samples = 0
    
    
    # calculates number of products that has not been bought / number of products bought
    # in all the baskets
    for batch_data in train_dataloader:
        _, _, _, targets, _ = batch_data
        total_positive_items += targets.sum().item()
        total_samples += targets.shape[0]

    
    num_products = model.num_products + 1
    num_negatives = (num_products * total_samples) - total_positive_items
    num_positives = total_positive_items
    pos_weight_value = num_negatives / num_positives
    
    pos_weight = torch.full((model.num_products + 1,), pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_hits = 0
        total_relevant = 0
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_data in progress_bar:
            products, categories, timestamps, targets, batch_size = batch_data
            
            products = products.to(device)
            categories = categories.to(device)
            timestamps = timestamps.to(device)
            targets = targets.to(device)
            
            logits = model(products, categories, timestamps, batch_size)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                _, top_indices = torch.topk(logits, K, dim=1)
                hits_tensor = torch.gather(targets, 1, top_indices)
                total_hits += hits_tensor.sum().item()
                total_relevant += targets.sum().item()
            
            if targets.sum().item() > 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'R@{K}': f'{hits_tensor.sum().item()/targets.sum().item():.2%}'
                })
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_recall = total_hits / total_relevant if total_relevant > 0 else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Recall@{K}: {train_recall:.2%}")
        
        
    return model

def collate_fn_with_category_lists(batch, max_basket_len, max_cats_per_product):
    """Collate function dla list list kategorii - z parametrami"""
    product_input, categories_input, user_timestamps, target_vecs = zip(*batch)
    
    target_vec = torch.stack(target_vecs)
    batch_size = len(batch)
                
    # === 2. Products padding ===
    flat_products = [] # befode [[basket1], [basket2]] -> [[basket1, basket2, ..]]
    for seq in product_input:
        flat_products.extend(seq)
    padded_products_flat = []
    for basket in flat_products:
        padded_basket = basket + [0] * (max_basket_len - len(basket))
        padded_products_flat.append(padded_basket)
    
    products_tensor = torch.tensor(padded_products_flat, dtype=torch.long)
    
    # === 3. Categories padding ===
    flat_categories = []
    for seq in categories_input:
        flat_categories.extend(seq)
    
    padded_categories_flat = []
    for basket_categories in flat_categories:
        basket_padded = []
        
        # Padding liczby produktów w koszyku
        while len(basket_categories) < max_basket_len:
            basket_categories.append([])  # Pusta lista dla brakującego produktu
        
        # Dla każdego produktu
        for product_cats in basket_categories[:max_basket_len]:
            if not isinstance(product_cats, list):
                product_cats = [product_cats] if product_cats != 0 else []
            
            if len(product_cats) < max_cats_per_product:
                padded_cats = product_cats + [0] * (max_cats_per_product - len(product_cats))
            basket_padded.append(padded_cats)
        
        padded_categories_flat.append(basket_padded)
    
    categories_tensor = torch.tensor(padded_categories_flat, dtype=torch.long)
    
    # === 4. Timestamps ===
    timestamps_tensor = torch.tensor(user_timestamps, dtype=torch.long)
    
    return (
        products_tensor,
        categories_tensor,
        timestamps_tensor,
        target_vec,
        batch_size,
    )

def test_model(model, test_dataloader, K, device):
    """Testowanie modelu na danych testowych"""
    model.eval()
    model.to(device)
    
    total_hits = 0
    total_relevant = 0
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc="Testowanie"):
            products, categories, timestamps, targets, batch_size = batch_data
            
            products = products.to(device)
            categories = categories.to(device)
            timestamps = timestamps.to(device)
            targets = targets.to(device)
            
            logits = model(products, categories, timestamps, batch_size)
            
            # Top-K predictions
            _, top_indices = torch.topk(logits, K, dim=1)
            hits = torch.gather(targets, 1, top_indices).sum() # takes in batch top indices then goes to target and checks wheater there is 1 and sum
            
            total_hits += hits.item()
            total_relevant += targets.sum().item()
    
    # Oblicz metryki
    recall = total_hits / total_relevant if total_relevant > 0 else 0
    
    print(f"\n📊 WYNIKI TESTOWE:")
    print(f"   Recall@{K}: {recall:.2%}")
    print(f"   Trafione produkty: {total_hits:,} / {total_relevant:,}")
    
    return recall

def get_prediction(user_histories, product_processor):
    """Główna funkcja kompatybilna z twoim kodem"""
    SEQ_LENGTH = 8
    BATCH_SIZE = 1024
    NUM_EPOCHS = 7
    K = 10
    PRODUCT_EMBEDDING_DIM = 128
    CATEGORY_EMBEDDING_DIM =64
    HIDDEN_DIM = 128
    NUM_LAYERS = 5
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    TIMESTAMP_EMBEDDING_DIM = 8


    MAX_BASKET_LEN = product_processor.max_basket_len
    MAX_CATS_PER_PRODUCT = product_processor.max_cats_per_product
    
    print("=" * 60)
    print("SYSTEM REKOMENDACJI Z LISTAMI LIST KATEGORII")
    print("=" * 60)
    
    # Stwórz dataset
    print("\nTworzenie datasetów...")
    user_ids = user_histories['user_id'].unique()
    train_user_ids, test_user_ids = train_test_split(user_ids, test_size=0.2, random_state=42)
    
    train_data = user_histories[user_histories['user_id'].isin(train_user_ids)]
    test_data = user_histories[user_histories['user_id'].isin(test_user_ids)]
    print(f"  Train: {len(train_data):,} wierszy ({len(train_user_ids)} użytkowników)")
    print(f"  Test: {len(test_data):,} wierszy ({len(test_user_ids)} użytkowników)")    
    
    print("\n📚 TWORZENIE DATASETU TRENINGOWEGO...")
    train_dataset = ProductSequenceDatasetWithCategories(train_data, product_processor, sequence_length=SEQ_LENGTH)
    print("\n🧪 TWORZENIE DATASETU TESTOWEGO...")
    test_dataset = ProductSequenceDatasetWithCategories(test_data, product_processor, sequence_length=SEQ_LENGTH)    
    
    num_categories = product_processor.get_num_categories()
    print(f"  Liczba produktów: {train_dataset.num_products}")
    print(f"  Liczba kategorii: {num_categories}")
    print(f"  Utworzono {len(train_dataset):,} sekwencji")
    
    collate_fn = partial(
        collate_fn_with_category_lists,
        max_basket_len=MAX_BASKET_LEN,
        max_cats_per_product=MAX_CATS_PER_PRODUCT
    )
    # Stwórz dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
)
    
    # Inicjalizuj model
    print("\nInicjalizacja modelu...")
    model = LSTMRecommenderWithCategoryMean(
        num_products=train_dataset.num_products,
        num_categories=num_categories,
        category_embedding_dim=CATEGORY_EMBEDDING_DIM,
        product_embedding_dim=PRODUCT_EMBEDDING_DIM,
        timestamp_embedding_dim=TIMESTAMP_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQ_LENGTH,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    print(f"  Parametry modelu: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trenuj model
    print("\n" + "=" * 60)
    print("ROZPOCZĘCIE TRENINGU")
    print("=" * 60)
    
    trained_model = train_model_with_categories(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        K=K
    )
    print("\n" + "=" * 60)
    print("TESTOWANIE NA ZBIORZE TESTOWYM")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_recall = test_model(trained_model, test_dataloader, K, device)

    # Zapisz model
    models_dir = Path(__file__).parent / "saved_models"
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    recall_str = f"{test_recall:.3f}"[2:]
    model_filename = f"lstm_cat_{timestamp}_rec{recall_str}.pth"
    model_path = models_dir / model_filename
    
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'num_products': train_dataset.num_products,
            'num_categories': num_categories,
            'product_embedding_dim': PRODUCT_EMBEDDING_DIM,
            'category_embedding_dim': CATEGORY_EMBEDDING_DIM,
            'timestamp_embedding_dim': TIMESTAMP_EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'sequence_length': SEQ_LENGTH,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        },
        'test_recall': test_recall
    }, model_path)
    
    print(f"\n✅ Model zapisany w: {model_path}")
    print(f"✅ Recall@{K} na testowym: {test_recall:.2%}")
    
    return trained_model
