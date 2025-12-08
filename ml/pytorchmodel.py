import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class LSTMRecommender(nn.Module):
    def __init__(self, num_products, num_categories, product_categories, 
                 product_embedding_dim=64, hidden_dim=128, sequence_length=4,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.num_products = num_products
        self.embedding_dim = product_embedding_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Embedding layers
        self.product_embedding = nn.Embedding(num_products + 1, product_embedding_dim, padding_idx=0)
        self.category_embedding_dim = product_embedding_dim // 2
        self.category_embedding = nn.Embedding(num_categories + 1, self.category_embedding_dim, padding_idx=0)
        
        # Timestamp embedding
        self.timestamp_dim = product_embedding_dim // 4
        self.timestamp = nn.Linear(1, self.timestamp_dim)
        
        # LSTM
        input_dim = product_embedding_dim + self.category_embedding_dim + self.timestamp_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_products + 1)
    
    def forward(self, products_tensor, categories_tensor, timestamps_tensor, max_basket_len, batch_size):
        device = self.product_embedding.weight.device
        
        # Move tensors to device
        products_tensor = products_tensor.to(device)
        categories_tensor = categories_tensor.to(device)
        timestamps_tensor = timestamps_tensor.to(device)
        
        # 1. Get embeddings
        product_embeds = self.product_embedding(products_tensor)
        category_embeds = self.category_embedding(categories_tensor)
        
        # 2. Create masks for padding
        product_mask = (products_tensor != 0).float().unsqueeze(-1)
        category_mask = (categories_tensor != 0).float().unsqueeze(-1)
        
        # 3. Calculate mean with masking (IGNORE PADDING!)
        sum_product_embeds = torch.sum(product_embeds * product_mask, dim=1)
        num_product_items = torch.clamp(torch.sum(product_mask, dim=1), min=1.0)
        product_embed_mean = sum_product_embeds / num_product_items
        
        sum_category_embeds = torch.sum(category_embeds * category_mask, dim=1)
        num_category_items = torch.clamp(torch.sum(category_mask, dim=1), min=1.0)
        category_embed_mean = sum_category_embeds / num_category_items
        
        # 4. Timestamp embeddings
        timestamps_flat = timestamps_tensor.view(-1, 1)
        timestamp_embed = self.timestamp(timestamps_flat)
        
        # 5. Combine embeddings
        combined_embed = torch.cat([
            product_embed_mean, 
            category_embed_mean, 
            timestamp_embed
        ], dim=1)
        
        # 6. Reshape for LSTM
        lstm_input = combined_embed.view(batch_size, self.sequence_length, -1)
        
        # 7. LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # 8. Get last output
        last_output = lstm_out[:, -1, :]
        
        # 9. Output layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class ProductSequenceDataset(Dataset):
    def __init__(self, user_histories, product_processor, sequence_length=4):
        self.user_histories = user_histories
        self.product_processor = product_processor
        self.sequence_length = sequence_length
        self.num_products = product_processor.get_vocab_size()
        
        self.input_basket_products_sequences = []  
        self.input_basket_categories_sequences = []
        self.target_baskets = []          
        self.user_timestamps = []         
        
        print(f"\n=== INFORMACJE O DATASET ===")
        print(f"Kolumny w danych: {list(user_histories.columns)}")
        print(f"Liczba wierszy: {len(user_histories):,}")
        
        # Sprawdź jakie kolumny faktycznie mamy
        self.has_order_number = 'order_number' in user_histories.columns
        self.has_days_since_prior = 'days_since_prior_order' in user_histories.columns
        
        print(f"Ma 'order_number': {self.has_order_number}")
        print(f"Ma 'days_since_prior_order': {self.has_days_since_prior}")
        
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        print("\nPrzygotowywanie sekwencji...")
        
        # Grupuj po user_id
        grouped_user_data = self.user_histories.groupby('user_id')
        total_users = self.user_histories['user_id'].nunique()
        
        print(f"Przetwarzanie {total_users:,} unikalnych użytkowników...")
        
        sequences_count = 0
        for user_id, user_data in tqdm(grouped_user_data, desc="Użytkownicy", total=total_users):
            # Jeśli mamy order_number, sortujemy po nim
            if self.has_order_number:
                user_data = user_data.sort_values('order_number')
            # W przeciwnym razie zakładamy, że dane są już posortowane
            
            # Tylko użytkownicy z wystarczającą liczbą zamówień
            if len(user_data) > self.sequence_length:
                for seq_start in range(len(user_data) - self.sequence_length):
                    # Input sequence
                    input_products = list(user_data['product_id'].iloc[seq_start:seq_start + self.sequence_length])
                    input_categories = list(user_data['aisle_id'].iloc[seq_start:seq_start + self.sequence_length])
                    
                    # Target basket
                    target_basket = list(user_data['product_id'].iloc[seq_start + self.sequence_length])
                    
                    # Timestamps
                    if self.has_days_since_prior:
                        timestamps = list(user_data['days_since_prior_order'].iloc[seq_start:seq_start + self.sequence_length])
                    else:
                        # Jeśli nie ma timestampów, użyj sekwencji [1, 2, 3, 4]
                        timestamps = list(range(1, self.sequence_length + 1))
                    
                    self.input_basket_products_sequences.append(input_products)
                    self.input_basket_categories_sequences.append(input_categories)
                    self.target_baskets.append(target_basket)
                    self.user_timestamps.append(timestamps)
                    
                    sequences_count += 1
        
        print(f"\nUtworzono {sequences_count:,} sekwencji")
        print(f"Przykładowa sekwencja:")
        if sequences_count > 0:
            print(f"  Input products: {self.input_basket_products_sequences[0]}")
            print(f"  Input categories: {self.input_basket_categories_sequences[0]}")
            print(f"  Target basket: {self.target_baskets[0]}")
            print(f"  Timestamps: {self.user_timestamps[0]}")
    
    def __len__(self):
        return len(self.input_basket_products_sequences)
    
    def __getitem__(self, idx):
        # Tworzenie target vector na żądanie
        target_basket = self.target_baskets[idx]
        target_vector = torch.zeros(self.num_products + 1, dtype=torch.float)
        
        for product_id in target_basket:
            if 0 < product_id <= self.num_products:
                target_vector[product_id] = 1.0
        
        return (
            self.input_basket_products_sequences[idx],  
            self.input_basket_categories_sequences[idx],
            torch.tensor(self.user_timestamps[idx], dtype=torch.float),
            target_vector
        )


def collate_fn_optimized(batch):
    """Collate function that handles variable-length baskets"""
    product_input, categories_input, user_timestamps, target_vecs = zip(*batch)
    
    # Stack target vectors
    target_vec = torch.stack(target_vecs)
    batch_size = len(batch)
    
    # Spłaszcz wszystkie koszyki w batchu
    flat_products = []
    flat_categories = []
    
    for prod_seq, cat_seq in zip(product_input, categories_input):
        flat_products.extend(prod_seq)
        flat_categories.extend(cat_seq)
    
    # Znajdź maksymalną długość koszyka
    max_basket_len = 0
    for basket in flat_products:
        max_basket_len = max(max_basket_len, len(basket))
    
    # Wyrównaj wszystkie koszyki do max_basket_len
    padded_products = []
    padded_categories = []
    
    for basket_prod, basket_cat in zip(flat_products, flat_categories):
        padded_prod = basket_prod + [0] * (max_basket_len - len(basket_prod))
        padded_cat = basket_cat + [0] * (max_basket_len - len(basket_cat))
        padded_products.append(padded_prod)
        padded_categories.append(padded_cat)
    
    # Create final tensors
    products_tensor = torch.tensor(padded_products, dtype=torch.long)
    categories_tensor = torch.tensor(padded_categories, dtype=torch.long)
    timestamps_tensor = torch.stack(user_timestamps)
    
    return (
        products_tensor,
        categories_tensor,
        timestamps_tensor,
        target_vec,
        max_basket_len,
        batch_size,
    )


def train_enhanced_model(model, train_dataloader, val_dataloader=None, 
                        num_epochs=5, learning_rate=0.001, K=10):
    """Uproszczona funkcja treningowa"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"\nTrening na: {device}")
    
    # Class weighting
    pos_weight_value = 50000 / 10 * 2  # 10,000
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
            products, categories, timestamps, targets, max_len, batch_size = batch_data
            
            # Move to device
            products = products.to(device)
            categories = categories.to(device)
            timestamps = timestamps.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(products, categories, timestamps, max_len, batch_size)
            
            # Calculate loss
            loss = criterion(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate training Recall@K
            with torch.no_grad():
                _, top_indices = torch.topk(logits, K, dim=1)
                hits_tensor = torch.gather(targets, 1, top_indices)
                total_hits += hits_tensor.sum().item()
                total_relevant += targets.sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'R@{K}': f'{hits_tensor.sum().item()/targets.sum().item():.2%}' if targets.sum().item() > 0 else '0%'
            })
        
        # Calculate training metrics
        avg_train_loss = total_loss / len(train_dataloader)
        train_recall = total_hits / total_relevant if total_relevant > 0 else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Recall@{K}: {train_recall:.2%}")
    
    return model


def show_example_predictions(model, dataset, product_processor, num_examples=3, K=10):
    """Pokazuj przykładowe predykcje z NAZWAMI produktów"""
    model.eval()
    device = next(model.parameters()).device
    
    for example_idx in range(min(num_examples, len(dataset))):
        print(f"\n{'='*50}")
        print(f"PRZYKŁAD {example_idx + 1}")
        print('='*50)
        
        # Get raw data
        input_products, input_categories, timestamps, target_vector = dataset[example_idx]
        true_target_basket = dataset.target_baskets[example_idx]
        
        # Display input sequence WITH PRODUCT NAMES
        print("\n📦 SEKWENCJA WEJŚCIOWA (ostatnie 4 koszyki):")
        for i, basket in enumerate(input_products):
            print(f"\n  Koszyk {i+1} ({len(basket)} produktów):")
            
            # Grupuj produkty po kategoriach dla lepszej czytelności
            category_groups = {}
            for prod_id, cat_id in zip(basket, input_categories[i]):
                if prod_id > 0:  # Ignoruj padding
                    cat_name = product_processor.get_category_name(cat_id)
                    prod_name = product_processor.get_product_name(prod_id)
                    
                    if cat_name not in category_groups:
                        category_groups[cat_name] = []
                    category_groups[cat_name].append(prod_name)
            
            # Wydrukuj zgrupowane
            for cat_name, products in category_groups.items():
                print(f"    [{cat_name.upper()}]: {', '.join(products[:3])}" + 
                      (f" + {len(products)-3} więcej" if len(products) > 3 else ""))
        
        # Display true target basket WITH NAMES
        print(f"\n🎯 PRAWDZIWY NASTĘPNY KOSZYK ({len(true_target_basket)} produktów):")
        for prod_id in true_target_basket[:10]:  # Pokaż max 10
            prod_name = product_processor.get_product_name(prod_id)
            print(f"  - {prod_name}")
        if len(true_target_basket) > 10:
            print(f"  ... i {len(true_target_basket) - 10} więcej")
        
        # Create batch for prediction
        batch = [(input_products, input_categories, timestamps, target_vector)]
        products_tensor, categories_tensor, timestamps_tensor, target_vec, max_basket_len, batch_size = collate_fn_optimized(batch)
        
        # Move to device
        products_tensor = products_tensor.to(device)
        categories_tensor = categories_tensor.to(device)
        timestamps_tensor = timestamps_tensor.to(device)
        
        # Get predictions
        with torch.no_grad():
            logits = model(products_tensor, categories_tensor, timestamps_tensor, max_basket_len, batch_size)
            probabilities = torch.sigmoid(logits)
            top_probs, top_indices = torch.topk(probabilities[0], K)
        
        # Display predictions WITH NAMES
        print(f"\n🏆 TOP-{K} REKOMENDACJI:")
        
        hits = 0
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            prod_name = product_processor.get_product_name(idx.item())
            prob_value = prob.item()
            is_correct = idx.item() in true_target_basket
            
            marker = " ✓" if is_correct else ""
            if is_correct:
                hits += 1
            
            # Formatowanie prawdopodobieństwa
            if prob_value > 0.9:
                prob_str = f"{prob_value:.3f} 🔥"
            elif prob_value > 0.5:
                prob_str = f"{prob_value:.3f}"
            else:
                prob_str = f"{prob_value:.3f} ⚠️"
            
            print(f"  {i+1:2d}. {prod_name:40} {prob_str}{marker}")
        
        print(f"\n📊 PODSUMOWANIE:")
        print(f"  Trafienia w Top-{K}: {hits}/{len(true_target_basket)}")
        print(f"  Recall@{K}: {hits/len(true_target_basket):.1%}")
        
        # Dodatkowa analiza
        print(f"\n🔍 ANALIZA:")
        if hits > 0:
            print(f"  ✅ Model trafił {hits} produktów")
        else:
            print(f"  ❌ Model nie trafił żadnego produktu")


def get_prediction(user_histories, product_processor):
    """Główna funkcja - kompatybilna z Twoim kodem"""
    SEQ_LENGTH = 4
    BATCH_SIZE = 4096
    NUM_EPOCHS = 2
    K = 10
    
    print("=" * 60)
    print("SYSTEM REKOMENDACJI - Next Basket Prediction")
    print("=" * 60)
    
    # Dataset statistics
    print(f"\nStatystyki datasetu:")
    print(f"  Liczba produktów: {product_processor.get_vocab_size():,}")
    print(f"  Liczba kategorii: {product_processor.get_num_categories():,}")
    
    # Sprawdź czy mamy dość danych dla walidacji
    if len(user_histories) < 10000:
        print("\nUWAGA: Mało danych, używam tylko train/test split")
        train_data = user_histories
        val_data = None
    else:
        # Split data
        print(f"\nDzielenie danych (Train/Test = 80%/20%)...")
        train_data, val_data = train_test_split(
            user_histories, test_size=0.2, random_state=42
        )
        print(f"  Training set: {len(train_data):,} rows")
        print(f"  Validation set: {len(val_data):,} rows")
    
    # Create datasets
    print("\nTworzenie datasetów...")
    train_dataset = ProductSequenceDataset(train_data, product_processor, sequence_length=SEQ_LENGTH)
    
    if val_data is not None:
        val_dataset = ProductSequenceDataset(val_data, product_processor, sequence_length=SEQ_LENGTH)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn_optimized
        )
    else:
        val_dataloader = None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn_optimized,
        num_workers=0  # Na Windows często lepiej 0
    )
    
    print(f"\nUtworzono {len(train_dataset):,} sekwencji treningowych")
    
    # Initialize model
    print("\nInicjalizacja modelu...")
    model = LSTMRecommender(
        num_products=product_processor.get_vocab_size(),
        num_categories=product_processor.get_num_categories(),
        product_categories=product_processor.product_categories,
        product_embedding_dim=64,
        hidden_dim=128,
        sequence_length=SEQ_LENGTH,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"  Parametry modelu: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "=" * 60)
    print("ROZPOCZĘCIE TRENINGU")
    print("=" * 60)
    
    trained_model = train_enhanced_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=0.001,
        K=K
    )
    
    # ZMIANA: Użyj show_example_predictions zamiast starego kodu
    print("\n" + "=" * 60)
    print("PRZYKŁADOWE PREDYKCJE Z NAZWAMI")
    print("=" * 60)
    
    show_example_predictions(
        model=trained_model,
        dataset=train_dataset,
        product_processor=product_processor,
        num_examples=3,
        K=10
    )
    
    return trained_model