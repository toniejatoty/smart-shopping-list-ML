import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class LSTMRecommender(nn.Module):
    def __init__(self, num_products, num_categories, product_categories, 
                 product_embedding_dim=64, hidden_dim=128, sequence_length=4,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.num_products = num_products
        self.embedding_dim = product_embedding_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        

        self.product_embedding = nn.Embedding(num_products + 1, product_embedding_dim, padding_idx=0)
        
        self.category_embedding_dim = product_embedding_dim // 2
        self.category_embedding = nn.Embedding(num_categories + 1, self.category_embedding_dim, padding_idx=0)
        
        self.timestamp_dim = product_embedding_dim // 4
        self.timestamp = nn.Linear(1, self.timestamp_dim)
        
        self.user_features_dim = product_embedding_dim // 4
        self.user_features_fc = nn.Linear(2, self.user_features_dim)

        input_dim = self.product_embedding.embedding_dim + self.category_embedding_dim + self.timestamp_dim + self.user_features_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_products+1 )
    
    def forward(self, product_input, categories_input, user_age_input, user_timestamps_input, user_gender_input):
        device = self.product_embedding.weight.device
        
        batch_sequences = []
        
        user_features = torch.stack([user_age_input, user_gender_input], dim=1).float().to(device)
        user_features_embed = self.user_features_fc(user_features) 


        for i, (user_products, user_categories, user_timestamps) in enumerate(zip(product_input, categories_input, user_timestamps_input)):
            user_embeddings = []
            
            for j, (basket,basket_categories, timestamp) in enumerate(zip(user_products, user_categories, user_timestamps)):
                
                product_tensor = torch.tensor(basket, dtype=torch.long, device=device)
                product_embeds = self.product_embedding(product_tensor)
                product_embed = torch.mean(product_embeds, dim=0)

                category_tensor = torch.tensor(basket_categories, dtype=torch.long, device=device)
                category_embeds = self.category_embedding(category_tensor)
                category_embed = torch.mean(category_embeds, dim=0)  

                timestamp_tensor = torch.tensor([timestamp], dtype=torch.float, device=device)
                timestamp_embed = self.timestamp(timestamp_tensor)  
            
                combined_embed = torch.cat([
                product_embed,           
                category_embed,          
                timestamp_embed,         
                user_features_embed[i]   ], dim=0)
                
                user_embeddings.append(combined_embed)
            
            user_sequence = torch.stack(user_embeddings)
            batch_sequences.append(user_sequence)
        
        baskets_tensor = torch.stack(batch_sequences)
        
        lstm_out, _ = self.lstm(baskets_tensor)
        
        last_output = lstm_out[:, -1, :]  
        
        x = self.dropout(last_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  
        
        return logits, None

class ProductSequenceDataset(Dataset):
    def __init__(self, user_histories, product_processor, sequence_length=4):
        self.user_histories = user_histories
        self.product_processor = product_processor
        self.sequence_length = sequence_length
        self.num_products = product_processor.get_vocab_size()
        
        self.input_basket_products_sequences = []  
        self.input_basket_categories_sequences = []
        self.target_baskets = []          
        self.target_vectors = []          
        self.user_ages = []               
        self.user_timestamps = []         
        self.user_gender = []
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        for user_id in set(self.user_histories['user_id']):
            user_data = self.user_histories[self.user_histories['user_id'] == user_id]
            
            for seq_start in range(len(user_data) - self.sequence_length ):
                
                input_products = list(user_data['products'].iloc[seq_start:seq_start + self.sequence_length])
                input_categories = list(user_data['categories'].iloc[seq_start:seq_start + self.sequence_length])
                target_basket = list(user_data['products'].iloc[seq_start + self.sequence_length])
                
                user_age = user_data['age'].values[0]
                user_gender = user_data['gender'].values[0]
                timestamps = list(user_data['timestamp'].iloc[seq_start:seq_start + self.sequence_length ])
                
                
                target_vector = torch.zeros(self.num_products+1)  
                for product_id in target_basket:
                    target_vector[product_id] = 1.0
                
                
                self.input_basket_products_sequences.append(input_products)
                self.input_basket_categories_sequences.append(input_categories)
                self.target_baskets.append(target_basket)  
                self.target_vectors.append(target_vector)
                self.user_ages.append(user_age)
                self.user_timestamps.append(timestamps)
                self.user_gender.append(user_gender)

    def __len__(self):
        return len(self.input_basket_products_sequences)
    
    def __getitem__(self, idx):
        return (
            self.input_basket_products_sequences[idx],  
            self.input_basket_categories_sequences[idx],
            torch.tensor(self.user_ages[idx], dtype=torch.float),
            torch.tensor(self.user_timestamps[idx], dtype=torch.float),
            torch.tensor(self.user_gender[idx], dtype=torch.float),
            torch.tensor(self.target_vectors[idx], dtype=torch.float)  
        )

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
            product_input, categories_input, user_age_input, user_timestamps_input, user_gender_input, target_vec = batch_data
            
            target_vec = target_vec.to(device)

            logits, attention_weights = model(product_input, categories_input, user_age_input, user_timestamps_input, user_gender_input)

            optimizer.zero_grad()
            
            loss = criterion(logits, target_vec)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                probabilities = torch.sigmoid(logits)
                predicted = (probabilities > 0.5).float()
                
                correct = (predicted == target_vec).sum().item()
                total_correct += correct
                total_elements += target_vec.numel()
                                
        epoch_accuracy = total_correct / total_elements
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
def get_prediction(user_histories, product_processor):
    # Ustawienia
    SEQ_LENGTH = 4
    BATCH_SIZE = 8
    NUM_EPOCHS = 100

    print(f"Liczba produktów: {product_processor.get_vocab_size()}")
    print(f"Liczba kategorii: {product_processor.get_num_categories()}")

    dataset = ProductSequenceDataset(
        user_histories, 
        product_processor, 
        sequence_length=SEQ_LENGTH
    )
    def simple_collate_fn(batch):
        product_input, categories_input, user_ages, user_timestamps, user_genders, target_vecs = zip(*batch)

        target_vec = torch.stack(target_vecs)
        return (
            product_input,    
            categories_input,  
            torch.stack(user_ages),       
            user_timestamps,   
            torch.stack(user_genders),       
            target_vec       
        )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=simple_collate_fn)

    print(f"Utworzono {len(dataset)} przykładów treningowych")

    model = LSTMRecommender(
        num_products=product_processor.get_vocab_size(),
        num_categories=product_processor.get_num_categories(),
        product_categories=product_processor.product_categories,
        product_embedding_dim=64,
        hidden_dim=128,
        sequence_length=SEQ_LENGTH,
    )

    print(f"Model ma {sum(p.numel() for p in model.parameters()):,} parametrów")
    print("Rozpoczynanie trenowania...")
    train_enhanced_model(model, dataloader, num_epochs=NUM_EPOCHS, learning_rate=0.001)

    print("\n--- Przykład predykcji ---")
    model.eval()
    with torch.no_grad():
        test_idx = 0
        product_input, categories_input, user_age, user_timestamps, user_gender, target_vec = dataset[test_idx]
        
        logits, attention_weights = model(
            [product_input], [categories_input], user_age.unsqueeze(0), [user_timestamps], user_gender.unsqueeze(0)
        )
        
        probabilities = torch.sigmoid(logits)
        
        top_probs, top_indices = torch.topk(probabilities, 10)
        
        print("Sekwencja wejściowa koszyków:")
        for i, basket in enumerate(dataset.input_basket_products_sequences[test_idx]):
            print(f"  Koszyk {i+1}: {[product_processor.id_to_product.get(product) for product in basket]}")

        print("Prawdziwy target koszyk:", [product_processor.id_to_product.get(product) for product in dataset.target_baskets[test_idx]])
        print("Top 10 rekomendowanych produktów:")
        
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            
            product_name = product_processor.id_to_product.get(idx.item(), f"Product_{idx.item()}")
            marker = " ✓" if idx.item() in dataset.target_baskets[test_idx] else ""
            print(f"{i+1}. {product_name}, p={prob.item():.4f}{marker}")
            
