import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerShoppingListRecommender(nn.Module):
    def __init__(self, num_products, num_categories, embedding_dim, hidden_dim, sequence_length, num_layers, dropout):
        super().__init__()
        
        self.sequence_length = sequence_length  # Zapisz dla encode_product
        
        self.product_embedding = nn.Embedding(
            num_products + 1, 
            embedding_dim, 
            padding_idx=0
        )
        
        self.category_embedding = nn.Embedding(
            num_categories + 1, 
            embedding_dim, 
            padding_idx=0
        )
        
        self.time_embedding = nn.Embedding(31, 16)
        
        combined_dim = (embedding_dim * 2) + 16
        self.input_projection = nn.Linear(combined_dim, hidden_dim)
        
        self.pos_embedding = nn.Embedding(sequence_length, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Post-norm: LayerNorm NA KOŃCU (bardziej stabilne)
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, products, categories, times):
        
        times = torch.clamp(times, max=30)
        
        product_emb = self.product_embedding(products)
        category_emb = self.category_embedding(categories)
        time_emb = self.time_embedding(times)

        cat_mask = (categories != 0).float().unsqueeze(-1)
        c_emb_masked = category_emb * cat_mask
        c_emb_mean = c_emb_masked.sum(dim=-2) / torch.clamp(cat_mask.sum(dim=-2), min=1.0)
        
        combined_items = torch.cat([product_emb, c_emb_mean], dim=-1)
        
        item_mask = (products != 0).float().unsqueeze(-1)
        
        basket_emb = (combined_items * item_mask).sum(dim=2) / torch.clamp(item_mask.sum(dim=2), min=1.0)
        
        x = torch.cat([basket_emb, time_emb], dim=-1)
        
        x = self.input_projection(x)
        
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        basket_mask = (products.sum(dim=-1) == 0)  
        
        x = self.transformer(x, src_key_padding_mask=basket_mask)
        
        x = self.final_norm(x[:, -1, :])
        x = self.fc_out(x)
        return x
    
    def encode_product(self, products, categories):

        product_emb = self.product_embedding(products)
        category_emb = self.category_embedding(categories)

        cat_mask = (categories != 0).float().unsqueeze(-1)
        c_emb_masked = category_emb * cat_mask
        c_emb_mean = c_emb_masked.sum(dim=-2) / torch.clamp(cat_mask.sum(dim=-2), min=1.0)

        combined_items = torch.cat([product_emb, c_emb_mean], dim=-1)

        item_mask = (products != 0).float().unsqueeze(-1)
        basket_emb = (combined_items * item_mask).sum(dim=1) / torch.clamp(item_mask.sum(dim=1), min=1.0)

        time_emb = self.time_embedding(torch.zeros(products.size(0), dtype=torch.long, device=products.device))
        x = torch.cat([basket_emb, time_emb], dim=-1)

        x = self.input_projection(x)

        x = x.unsqueeze(1) 
        positions = torch.full((x.size(0), 1), self.sequence_length - 1, dtype=torch.long, device=x.device)
        x = x + self.pos_embedding(positions)

        basket_mask = (products.sum(dim=-1) == 0).unsqueeze(1)
        x = self.transformer(x, src_key_padding_mask=basket_mask)

        x = self.final_norm(x[:, 0, :])
        x = self.fc_out(x)
        return x