# pytorchmodel_with_categories.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import datetime
from pathlib import Path
warnings.filterwarnings('ignore')


class LSTMRecommenderWithCategoryMean(nn.Module):
    def __init__(self, num_products, num_categories, 
                 category_embedding_dim=32, 
                 product_embedding_dim=64, 
                 timestamp_embedding_dim=8,
                 hidden_dim=128, 
                 sequence_length=4, 
                 num_layers=2, 
                 dropout=0.3):
        super().__init__()
        self.num_products = num_products
        self.num_categories = num_categories
        self.sequence_length = sequence_length
        
        # Embeddingi
        self.product_embedding = nn.Embedding(num_products + 1, product_embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, category_embedding_dim, padding_idx=0)
        
        # Timestamp
        self.timestamp = nn.Linear(1, timestamp_embedding_dim)
        
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
        """
        
        batch_size_seq, max_basket_len = products_tensor.shape[:2]
        
        # 1. Product embeddings
        product_embeds = self.product_embedding(products_tensor)
        
        # 2. Kategorie: oblicz mean embedding dla każdego produktu
        batch_size_seq, basket_size, max_cats = categories_tensor.shape
        
        # Pobierz embeddingi dla WSZYSTKICH kategorii
        flat_categories = categories_tensor.view(-1)
        cat_embeddings_all = self.category_embedding(flat_categories)
        
        # Reshape z powrotem
        cat_embeddings = cat_embeddings_all.view(batch_size_seq, basket_size, max_cats, -1)
        
        # Mask: które kategorie są prawdziwe (nie padding)
        cat_mask = (categories_tensor != 0).float().unsqueeze(-1)
        
        # Mean kategorii per produkt (ignoruj padding)
        sum_cat_per_product = torch.sum(cat_embeddings * cat_mask, dim=2)
        num_cats_per_product = torch.clamp(torch.sum(cat_mask, dim=2), min=1.0)
        mean_cat_per_product = sum_cat_per_product / num_cats_per_product
        
        # 3. Mean produktów w koszyku (ignoruj padding)
        product_mask = (products_tensor != 0).float().unsqueeze(-1)
        
        # Suma embeddingów produktów
        sum_product_embeds = torch.sum(product_embeds * product_mask, dim=1)
        num_products_in_basket = torch.clamp(torch.sum(product_mask, dim=1), min=1.0)
        product_mean = sum_product_embeds / num_products_in_basket
        
        # 4. Suma mean kategorii per koszyk
        sum_cat_embeds = torch.sum(mean_cat_per_product * product_mask, dim=1)
        category_mean = sum_cat_embeds / num_products_in_basket
        
        # 5. Timestamp embeddings
        timestamps_flat = timestamps_tensor.view(-1, 1)
        timestamp_embed = self.timestamp(timestamps_flat)
        
        # 6. Połącz wszystkie embeddings
        combined_embed = torch.cat([product_mean, category_mean, timestamp_embed], dim=1)
        
        # 7. Reshape dla LSTM: [batch, seq, features]
        lstm_input = combined_embed.view(batch_size, self.sequence_length, -1)
        
        # 8. LSTM
        lstm_out, _ = self.lstm(lstm_input)
        last_output = lstm_out[:, -1, :]
        
        # 9. Output
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class ProductSequenceDatasetWithCategories(Dataset):
    def __init__(self, user_histories, product_processor, sequence_length=4):
        self.user_histories = user_histories
        self.product_processor = product_processor
        self.sequence_length = sequence_length
        self.num_products = product_processor.get_vocab_size()
        
        # Słownik nazw kategorii -> ID
        self.category_name_to_id = {}
        self.id_to_category_name = {}
        self.next_category_id = 1
        self._build_category_vocab(user_histories)
        
        self.input_basket_products_sequences = []  
        self.input_basket_categories_sequences = []
        self.target_baskets = []          
        self.user_timestamps = []         
        
        print(f"\n=== INFORMACJE O DATASET Z KATEGORIAMI ===")
        print(f"Kolumny w danych: {list(user_histories.columns)}")
        print(f"Liczba wierszy: {len(user_histories):,}")
        print(f"Unikalnych nazw kategorii: {len(self.category_name_to_id)}")
        
        self.has_order_number = 'order_number' in user_histories.columns
        self.has_days_since_prior = 'days_since_prior_order' in user_histories.columns
        
        self._prepare_sequences()
    
    def _build_category_vocab(self, user_histories):
        """Zbuduj słownik nazw kategorii z list list"""
        print("Budowanie słownika kategorii z list list...")
        
        all_category_names = set()
        
        # Iteruj przez wszystkie listy list kategorii
        for categories_list in user_histories['aisle_id']:
            if isinstance(categories_list, list):
                for product_categories in categories_list:
                    if isinstance(product_categories, list):
                        for cat_name in product_categories:
                            if cat_name and str(cat_name) != 'nan':  # niepusta nazwa
                                all_category_names.add(str(cat_name))
        
        # Dodaj "unknown" dla padding
        self.category_name_to_id["[PAD]"] = 0
        self.id_to_category_name[0] = "[PAD]"
        
        # Dodaj wszystkie unikalne kategorie
        for cat_name in sorted(all_category_names):
            self.category_name_to_id[cat_name] = self.next_category_id
            self.id_to_category_name[self.next_category_id] = cat_name
            self.next_category_id += 1
        
        print(f"  Załadowano {len(self.category_name_to_id)} unikalnych kategorii")
    
    def _prepare_sequences(self):
        print("\nPrzygotowywanie sekwencji z listami list kategorii...")
        
        grouped_user_data = self.user_histories.groupby('user_id')
        total_users = self.user_histories['user_id'].nunique()
        
        print(f"Przetwarzanie {total_users:,} unikalnych użytkowników...")
        
        sequences_count = 0
        for user_id, user_data in tqdm(grouped_user_data, desc="Użytkownicy", total=total_users):
            if self.has_order_number:
                user_data = user_data.sort_values('order_number')
            
            if len(user_data) > self.sequence_length:
                for seq_start in range(len(user_data) - self.sequence_length):
                    # Input products
                    input_products = list(user_data['off_product_id'].iloc[seq_start:seq_start + self.sequence_length])
                    
                    # Input categories: KONWERTUJ listy list
                    input_categories = []
                    raw_categories = list(user_data['aisle_id'].iloc[seq_start:seq_start + self.sequence_length])
                    
                    for basket_categories in raw_categories:
                        basket_category_ids = []
                        
                        if isinstance(basket_categories, list):
                            for product_categories in basket_categories:
                                if isinstance(product_categories, list) and product_categories:
                                    # Konwertuj nazwy kategorii na ID
                                    cat_ids = [self.category_name_to_id.get(str(cat), 0) for cat in product_categories]
                                    basket_category_ids.append(cat_ids)
                                else:
                                    basket_category_ids.append([0])
                        else:
                            basket_category_ids.append([0])
                        
                        input_categories.append(basket_category_ids)
                    
                    # Target basket
                    target_basket = list(user_data['off_product_id'].iloc[seq_start + self.sequence_length])
                    
                    # Timestamps
                    if self.has_days_since_prior:
                        timestamps = list(user_data['days_since_prior_order'].iloc[seq_start:seq_start + self.sequence_length])
                    else:
                        timestamps = list(range(1, self.sequence_length + 1))
                    
                    self.input_basket_products_sequences.append(input_products)
                    self.input_basket_categories_sequences.append(input_categories)
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
            if 0 < off_product_id <= self.num_products:
                target_vector[off_product_id] = 1.0
        
        return (
            self.input_basket_products_sequences[idx],  
            self.input_basket_categories_sequences[idx],
            torch.tensor(self.user_timestamps[idx], dtype=torch.float),
            target_vector
        )
    
    def get_num_categories(self):
        return len(self.category_name_to_id)


def collate_fn_with_category_lists(batch):
    """Collate function dla list list kategorii"""
    product_input, categories_input, user_timestamps, target_vecs = zip(*batch)
    
    target_vec = torch.stack(target_vecs)
    batch_size = len(batch)
    
    # Spłaszcz produkty
    flat_products = []
    for prod_seq in product_input:
        flat_products.extend(prod_seq)
    
    # Spłaszcz kategorie
    flat_categories = []
    for cat_seq in categories_input:
        for basket_categories in cat_seq:
            flat_categories.append(basket_categories)
    
    # 1. Maksymalna długość koszyka
    max_basket_len = 0
    for basket in flat_products:
        max_basket_len = max(max_basket_len, len(basket))
    
    # 2. Maksymalna liczba kategorii per produkt
    max_categories_per_product = 0
    for basket_categories in flat_categories:
        for product_categories in basket_categories:
            max_categories_per_product = max(max_categories_per_product, len(product_categories))
    
    # 3. Wyrównaj produkty
    padded_products = []
    for basket_prod in flat_products:
        padded_prod = basket_prod + [0] * (max_basket_len - len(basket_prod))
        padded_products.append(padded_prod)
    
    # 4. Wyrównaj kategorie
    padded_categories = []
    for basket_categories in flat_categories:
        basket_padded = []
        
        # Wyrównaj liczbę produktów w koszyku
        while len(basket_categories) < max_basket_len:
            basket_categories.append([0])
        
        # Dla każdego produktu
        for product_categories in basket_categories[:max_basket_len]:
            if len(product_categories) < max_categories_per_product:
                padded_cats = product_categories + [0] * (max_categories_per_product - len(product_categories))
            else:
                padded_cats = product_categories[:max_categories_per_product]
            basket_padded.append(padded_cats)
        
        padded_categories.append(basket_padded)
    
    # 5. Utwórz tensory
    products_tensor = torch.tensor(padded_products, dtype=torch.long)
    categories_tensor = torch.tensor(padded_categories, dtype=torch.long)
    timestamps_tensor = torch.stack(user_timestamps)
    
    return (
        products_tensor,
        categories_tensor,
        timestamps_tensor,
        target_vec,
        batch_size,
    )


def train_model_with_categories(model, train_dataloader, val_dataloader=None, 
                               num_epochs=5, learning_rate=0.001, K=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"\nTrening na: {device}")
    
    # Oblicz pos_weight
    total_positive_items = 0
    total_samples = 0
    
    for batch_data in train_dataloader:
        _, _, _, targets, _ = batch_data
        total_positive_items += targets.sum().item()
        total_samples += targets.shape[0]
    
    num_products = model.num_products + 1
    num_negatives = (num_products * total_samples) - total_positive_items
    num_positives = total_positive_items
    pos_weight_value = num_negatives / max(num_positives, 1)
    
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
        
        if val_dataloader is not None:
            val_recall = validate_model(model, val_dataloader, K=K, device=device)
            print(f"  [VAL] Recall@{K}: {val_recall:.2%}")
    
    return model


def validate_model(model, val_dataloader, K=10, device='cuda'):
    model.eval()
    total_hits = 0
    total_relevant = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            products, categories, timestamps, targets, batch_size = batch
            
            products = products.to(device)
            categories = categories.to(device)
            timestamps = timestamps.to(device)
            targets = targets.to(device)
            
            logits = model(products, categories, timestamps, batch_size)
            
            _, top_indices = torch.topk(logits, K, dim=1)
            hits = torch.gather(targets, 1, top_indices).sum()
            total_hits += hits.item()
            total_relevant += targets.sum().item()
    
    return total_hits / total_relevant if total_relevant > 0 else 0


def get_prediction(user_histories, product_processor):
    """Główna funkcja kompatybilna z twoim kodem"""
    SEQ_LENGTH = 1#8
    BATCH_SIZE = 4096
    NUM_EPOCHS = 1#5
    K = 10
    PRODUCT_EMBEDDING_DIM = 1#128
    CATEGORY_EMBEDDING_DIM =1 #64
    HIDDEN_DIM = 1 #128
    NUM_LAYERS = 1 #2
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    TIMESTAMP_EMBEDDING_DIM = 1#8
    
    print("=" * 60)
    print("SYSTEM REKOMENDACJI Z LISTAMI LIST KATEGORII")
    print("=" * 60)
    
    # Podziel dane
    if len(user_histories) < 10000:
        train_data = user_histories
        val_data = None
    else:
        train_data, val_data = train_test_split(
            user_histories, test_size=0.2, random_state=42
        )
        print(f"Training set: {len(train_data):,} rows")
        print(f"Validation set: {len(val_data):,} rows")
    
    # Stwórz dataset
    print("\nTworzenie datasetu...")
    train_dataset = ProductSequenceDatasetWithCategories(train_data, product_processor, sequence_length=SEQ_LENGTH)
    num_categories = train_dataset.get_num_categories()
    
    print(f"  Liczba produktów: {train_dataset.num_products}")
    print(f"  Liczba kategorii: {num_categories}")
    print(f"  Utworzono {len(train_dataset):,} sekwencji")
    
    if val_data is not None:
        val_dataset = ProductSequenceDatasetWithCategories(val_data, product_processor, sequence_length=SEQ_LENGTH)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn_with_category_lists
        )
    else:
        val_dataloader = None
    
    # Stwórz dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn_with_category_lists,
        num_workers=0
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
        val_dataloader=val_dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        K=K
    )
    
    # Zapisz model
    models_dir = Path(__file__).parent / "saved_models"
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"lstm_with_categories_{timestamp}.pth"
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
        'category_vocab': train_dataset.category_name_to_id,
    }, model_path)
    
    print(f"\n✅ Model zapisany w: {model_path}")
    
    return trained_model