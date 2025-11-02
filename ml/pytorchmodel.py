import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import pandas as pd

class ProductProcessor:
    """Klasa zajmuje się przetworzeniem id produktu i kategorii na mniejszą liczbę. Oraz mapowanie
     produkt na kategorie jakie ma """
    def __init__(self):
        self.product_to_id = {}
        self.id_to_product = {}
        self.category_to_id = {}
        self.id_to_category = {}
        self.product_categories = {}  # Mapowanie product_id -> category_ids
        self.next_product_id = 1
        self.next_category_id = 1
    
    def process_product(self, product_data):
        """Przetwarza pojedynczy produkt i zwraca jego ID"""
        code = product_data.get('code', '') or product_data.get('_id', '')
        
        if code not in self.product_to_id:
            self.product_to_id[code] = self.next_product_id
            self.id_to_product[self.next_product_id] = {
                'code': code,
                'name': product_data.get('product_name', ''),
                'categories': product_data.get('categories', '')
            }
            
            # Przetwarzanie kategorii
            categories_str = product_data.get('categories', '')
            category_ids = self._process_categories(categories_str)
            self.product_categories[self.next_product_id] = category_ids
            
            self.next_product_id += 1
    
    def _process_categories(self, categories_str):
        """Przetwarza string kategorii na listę ID kategorii"""
        if not categories_str:
            return []
        
        # Zakładamy format "Kategoria1, Kategoria2, Kategoria3"
        categories = [cat.strip() for cat in categories_str.split(',')]
        category_ids = []
        
        for category in categories:
            if category and category not in self.category_to_id:
                self.category_to_id[category] = self.next_category_id
                self.id_to_category[self.next_category_id] = category
                self.next_category_id += 1
            if category:
                category_ids.append(self.category_to_id[category])
        return category_ids
    
    def get_vocab_size(self):
        return len(self.product_to_id)
    
    def get_num_categories(self):
        return len(self.category_to_id)

class EnhancedLSTMAttentionRecommender(nn.Module):
    def __init__(self, num_products, num_categories,product_categories, embedding_dim=128, hidden_dim=256, 
                 category_embedding_dim=32, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.num_products = num_products
        self.num_categories = num_categories
        self.product_categories = product_categories
        self.embedding_dim = embedding_dim
        self.category_embedding_dim = category_embedding_dim
        
        # Embedding dla produktów
        self.product_embedding = nn.Embedding(num_products + 1, embedding_dim, padding_idx=0)
        
        # Embedding dla kategorii
        self.category_embedding = nn.Embedding(num_categories + 1, category_embedding_dim, padding_idx=0)
        
        # LSTM - input: embedding_dim + category_embedding_dim
        combined_embedding_dim = embedding_dim + category_embedding_dim
        self.lstm = nn.LSTM(combined_embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Warstwy wyjściowe
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_products + 1)  # +1 dla paddingu
    
    def forward(self, product_sequences, sequence_lengths=None):
        batch_size, seq_len, basket_size = product_sequences.size()
        
        # Spłaszczamy do (batch_size * seq_len, basket_size) dla łatwiejszego przetwarzania
        product_sequences_flat = product_sequences.view(-1, basket_size)
        
        # 1. Embedding produktów - uśredniamy produkty w koszyku
        product_embedded_flat = self.product_embedding(product_sequences_flat)  # (batch_size*seq_len, basket_size, embedding_dim)
        product_embedded_avg = torch.mean(product_embedded_flat, dim=1)  # (batch_size*seq_len, embedding_dim)
        product_embedded = product_embedded_avg.view(batch_size, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
        
        # 2. Pobieramy kategorie dla produktów
        # Używamy pierwszego nie-padding produktu w koszyku, lub pierwszego produktu jeśli wszystkie są padding
        first_nonzero_products = []
        for batch_idx in range(batch_size):
            batch_products = []
            for seq_idx in range(seq_len):
                basket = product_sequences[batch_idx, seq_idx]
                # Szukamy pierwszego nie-padding produktu w koszyku
                non_zero_indices = torch.nonzero(basket, as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    first_product = basket[non_zero_indices[0]].item()
                else:
                    first_product = 0  # wszystkie produkty to padding
                batch_products.append(first_product)
            first_nonzero_products.append(batch_products)
        
        # 3. Embedding kategorii
        category_embedded_list = []
        for batch_idx in range(batch_size):
            seq_categories = []
            for seq_idx in range(seq_len):
                product_id = first_nonzero_products[batch_idx][seq_idx]
                if product_id == 0:  # padding
                    categories = [0]
                else:
                    categories = self.product_categories.get(product_id, [0])
                
                valid_categories = [cat for cat in categories if cat != 0]
                
                if valid_categories:
                    cat_tensor = torch.tensor(valid_categories, device=product_sequences.device)
                    cat_emb = self.category_embedding(cat_tensor)  # (num_cats, category_embedding_dim)
                    avg_cat_emb = torch.mean(cat_emb, dim=0)  # Uśredniamy kategorie
                else:
                    avg_cat_emb = torch.zeros(self.category_embedding_dim, device=product_sequences.device)
                
                seq_categories.append(avg_cat_emb)
            
            category_batch = torch.stack(seq_categories)  # (seq_len, category_embedding_dim)
            category_embedded_list.append(category_batch)
        
        category_embedded = torch.stack(category_embedded_list)  # (batch_size, seq_len, category_embedding_dim)
        
        # 4. Łączymy embeddingi produktów i kategorii
        combined_embedded = torch.cat([product_embedded, category_embedded], dim=2)  # (batch_size, seq_len, emb_dim+cat_emb_dim)
        
        # 5. LSTM
        if sequence_lengths is not None:
            combined_embedded = nn.utils.rnn.pack_padded_sequence(
                combined_embedded, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_output, (hidden, cell) = self.lstm(combined_embedded)
        
        if sequence_lengths is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # 6. Attention
        attention_scores = self.attention(lstm_output).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        context_vector = torch.sum(lstm_output * attention_weights, dim=1)  # (batch_size, hidden_dim)
        
        # 7. Warstwy wyjściowe
        x = self.dropout(context_vector)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  # (batch_size, num_products + 1)
        
        return logits, attention_weights.squeeze(-1)

class ProductSequenceDataset(Dataset):
    def __init__(self, user_histories, product_processor, sequence_length=4, max_basket_size=20):
        self.user_histories = user_histories
        self.product_processor = product_processor
        self.sequence_length = sequence_length
        self.max_basket_size = max_basket_size
        self.num_products = product_processor.get_vocab_size()
        
        self.input_basket_sequences = []  # Lista SEKWENCJI KOSZYKÓW
        self.target_baskets = []          # Pojedyncze koszyki target
        self.target_vectors = []          # Wektory binarne targetów
        
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        for user_history in self.user_histories:
            # user_history to lista koszyków: [basket1, basket2, basket3, ...]
            
            for i in range(len(user_history) - self.sequence_length):
                # Input: sekwencja koszyków (nie spłaszczona!)
                input_baskets = user_history[i:i + self.sequence_length]  # N koszyków
                
                # Target: następny koszyk jako lista produktów
                target_basket = user_history[i + self.sequence_length]
                
                # Target jako wektor binarny
                target_vector = torch.zeros(self.num_products + 1)  # +1 dla paddingu
                for product_code in target_basket:
                    product_id = self.product_processor.product_to_id[product_code]
                    target_vector[product_id] = 1.0
                
                self.input_basket_sequences.append(input_baskets)
                self.target_baskets.append(target_basket)  # Zachowujemy też oryginalny koszyk
                self.target_vectors.append(target_vector)
    
    def __len__(self):
        return len(self.input_basket_sequences)
    
    def __getitem__(self, idx):
        input_baskets = self.input_basket_sequences[idx]  # Lista N koszyków
        target_vec = self.target_vectors[idx]
        
        # Przygotowujemy tensory - ZACHOWUJĄC strukturę koszyków
        padded_baskets = []
        
        for basket in input_baskets:
            # Konwertujemy kody produktów na ID
            basket_ids = []
            for product_code in basket:
                basket_ids.append(self.product_processor.product_to_id[product_code])
            
            # Padding każdego koszyka do max_basket_size
            if len(basket_ids) < self.max_basket_size:
                padded_basket = basket_ids + [0] * (self.max_basket_size - len(basket_ids))
            else:
                padded_basket = basket_ids[:self.max_basket_size]
            padded_baskets.append(padded_basket)
        
        # Tensor: [sequence_length, max_basket_size]
        baskets_tensor = torch.tensor(padded_baskets, dtype=torch.long)
        
        return baskets_tensor, target_vec

def generate_sample_data_in_your_format(num_users=1000, num_products=500):
    """Generuje dane w formacie podobnym do Twojego"""
    product_processor = ProductProcessor()
    
    # Najpierw tworzymy produkty
    sample_products = []
    for i in range(1, num_products + 1):
        product = {
            '_id': f"59008200{1000 + i:04d}",
            'categories': "Nabiał, Mleko, Mleko bez laktozy" if i % 5 == 0 else 
                         "Pieczywo, Chleb" if i % 5 == 1 else
                         "Napoje, Kawa" if i % 5 == 2 else
                         "Słodycze, Czekolada" if i % 5 == 3 else
                         "Warzywa, Pomidory",
            'product_name': f"Product_{i}",
            'code': f"59008200{1000 + i:04d}"
        }
        sample_products.append(product)
        product_processor.process_product(product)
    
    # Teraz generujemy historie użytkowników
    user_histories = []
    for _ in range(num_users):
        num_baskets = np.random.randint(5, 15)
        user_history = []
        
        for _ in range(num_baskets):
            basket_size = np.random.randint(3, 8)
            basket_products = np.random.choice([p['code'] for p in sample_products], 
                                             basket_size, replace=False).tolist()
            user_history.append(basket_products)
        
        user_histories.append(user_history)
    
    return user_histories, product_processor

def train_enhanced_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_elements = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            baskets_tensor, target_vec = batch_data
            baskets_tensor = baskets_tensor.to(device)
            target_vec = target_vec.to(device)
            
            # FORWARD PASS
            logits, attention_weights = model(baskets_tensor)
            
            optimizer.zero_grad()
            
            # LOSS - multi-label
            loss = criterion(logits, target_vec)
            
            # BACKWARD PASS
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # POPRAWNE LICZENIE ACCURACY DLA MULTI-LABEL
            with torch.no_grad():
                probabilities = torch.sigmoid(logits)
                predicted = (probabilities > 0.5).float()
                
                # Accuracy: ile produktów dobrze przewidzianych
                correct = (predicted == target_vec).sum().item()
                total_correct += correct
                total_elements += target_vec.numel()
                
            if batch_idx % 10 == 0:
                batch_accuracy = correct / target_vec.numel()
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Acc: {batch_accuracy:.4f}')
                
        epoch_accuracy = total_correct / total_elements
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    return device

# Ustawienia
SEQ_LENGTH = 4
BATCH_SIZE = 32
MAX_BASKET_SIZE = 10
NUM_EPOCHS = 100

print("Generuje dane")
user_histories, product_processor = generate_sample_data_in_your_format(num_users=100, num_products=200)

print(f"Liczba produktów: {product_processor.get_vocab_size()}")
print(f"Liczba kategorii: {product_processor.get_num_categories()}")

dataset = ProductSequenceDataset(
    user_histories, 
    product_processor, 
    sequence_length=SEQ_LENGTH,
    max_basket_size=MAX_BASKET_SIZE
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Utworzono {len(dataset)} przykładów treningowych")

model = EnhancedLSTMAttentionRecommender(
    num_products=product_processor.get_vocab_size(),
    num_categories=product_processor.get_num_categories(),
    product_categories=product_processor.product_categories,
    embedding_dim=64,
    hidden_dim=128,
    category_embedding_dim=16
)

# Dodajemy product_categories do modelu

print(f"Model ma {sum(p.numel() for p in model.parameters()):,} parametrów")
print("Rozpoczynanie trenowania...")
device = train_enhanced_model(model, dataloader, num_epochs=NUM_EPOCHS, learning_rate=0.001)

# Przykład predykcji
print("\n--- Przykład predykcji ---")
model.eval()
with torch.no_grad():
    test_idx = 0
    baskets_tensor, target_vec = dataset[test_idx]
    baskets_tensor = baskets_tensor.unsqueeze(0).to(device)
    
    logits, attention_weights = model(baskets_tensor)
    probabilities = torch.sigmoid(logits)
    
    # Pobieramy top 10 produktów
    top_probs, top_indices = torch.topk(probabilities, 10)
    
    print("Sekwencja wejściowa koszyków:")
    for i, basket in enumerate(dataset.input_basket_sequences[test_idx]):
        print(f"  Koszyk {i+1}: {basket}")
    
    print("Prawdziwy target koszyk:", dataset.target_baskets[test_idx])
    print("Top 10 rekomendowanych produktów:")
    
    # Pobieramy prawdziwe ID produktów z target koszyka
    true_product_ids = []
    for product_code in dataset.target_baskets[test_idx]:
        product_id = product_processor.product_to_id[product_code]
        true_product_ids.append(product_id)
    
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        if idx.item() in product_processor.id_to_product:
            product_info = product_processor.id_to_product[idx.item()]
            marker = " ✓" if idx.item() in true_product_ids else ""

            print(f"{i+1}.{product_processor.id_to_product[int(idx.item())]}, p={prob.item():.4f} {marker}")
        else:
            marker = " ✓" if idx.item() in true_product_ids else ""
            print(f"{i+1}. Product_ID_{idx.item()} (p={prob.item():.4f}){marker}")

# print("\nUwagi dotyczące modelu:")
# print("1. Model przewiduje prawdopodobieństwo zakupu każdego produktu")
# print("2. ✓ oznacza produkty, które rzeczywiście były w target koszyku")
# print("3. Accuracy pokazuje jak dobrze model przewiduje wszystkie produkty (multi-label)")
# print("4. Loss maleje, a accuracy rośnie - model się uczy!")