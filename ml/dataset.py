from torch.utils.data import Dataset
import torch
import random

class ShoppingDataset(Dataset):
    def __init__(self, user_groups, seq_len=8, max_basket_size=100, max_cat_size=20, max_samples_per_user=None):
        self.seq_len = seq_len
        self.max_basket = max_basket_size
        self.max_cat = max_cat_size
        self.samples = []
        
        for user_history in user_groups:
            if len(user_history) >= seq_len + 1:
                # Wszystkie możliwe pozycje
                possible_starts = list(range(len(user_history) - seq_len))
                
                # OPTYMALIZACJA: Ogranicz liczbę sampli per user
                if max_samples_per_user and len(possible_starts) > max_samples_per_user:
                    # Wybierz równomiernie rozłożone pozycje
                    step = len(possible_starts) // max_samples_per_user
                    possible_starts = possible_starts[::step][:max_samples_per_user]
                
                for i in possible_starts:
                    self.samples.append((user_history, i))

    def __len__(self):
        return len(self.samples)

    def pad_list(self, items, max_len):
        return items[:max_len] + [0] * (max_len - len(items))

    def __getitem__(self, idx):
        user_history, start_idx = self.samples[idx]
        window = user_history.iloc[start_idx:start_idx + self.seq_len + 1]
        
        inputs = window.iloc[:-1]
        target = window.iloc[-1]

        in_products = []
        in_categories = []
        in_times = []

        num_padding = self.seq_len - len(inputs)
        for _ in range(num_padding):
            in_products.append([0] * self.max_basket)
            in_categories.append([[0] * self.max_cat] * self.max_basket)
            in_times.append(0) 

        for _, row in inputs.iterrows():
            in_products.append(self.pad_list(row['off_product_id'], self.max_basket))
            
            cats_in_basket = [self.pad_list(c, self.max_cat) for c in row['mapped_categories']]
            while len(cats_in_basket) < self.max_basket:
                cats_in_basket.append([0] * self.max_cat)
            in_categories.append(cats_in_basket[:self.max_basket])
            in_times.append(int(row['days_since_prior_order']))

        target_products = self.pad_list(target['off_product_id'], self.max_basket)
        target_categories = [self.pad_list(c, self.max_cat) for c in target['mapped_categories']]
        while len(target_categories) < self.max_basket:
            target_categories.append([0] * self.max_cat)

        return (
            torch.tensor(in_products), 
            torch.tensor(in_categories), 
            torch.tensor(in_times).long(),
            torch.tensor(target_products),
            torch.tensor(target_categories)[:self.max_basket]
        )