from fastapi import FastAPI
import pandas as pd
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
import numpy as np
from datetime import datetime, timedelta

class ShoppingRecommendationSystem:
    def __init__(self, products_data):
        self.simple_model = SimpleRecommendationModel()
        self.advanced_model = ImprovedRecommendationModel()
        self.users = pd.DataFrame(columns=['user_id', 'age', 'gender'])
        self.transactions = pd.DataFrame(columns=['user_id', 'product_id', 'timestamp'])
        self.products = pd.DataFrame(columns=['product_id', 'category', 'product_name'])
        self._needs_retraining = False
        self._initialize_products(products_data)
        
    def _initialize_products(self, products_data):
        products_list = []
        for product_id, (product_name, category) in enumerate(products_data.items()):
            products_list.append({
                'product_id': product_id,
                'category': category,
                'product_name': product_name
            })
        
        self.products = pd.DataFrame(products_list)
    def add_purchase(self, user_id, products, age, gender,time):
        """Dodaje nowy zakup do systemu"""
        
        # Aktualizuj dane użytkownika jeśli podano

        if user_id not in self.users['user_id'].values:
            new_user = {'user_id': user_id, 'age': age, 'gender': gender}
            self.users = pd.concat([self.users, pd.DataFrame([new_user])], ignore_index=True)
        

        for product in products:
            new_transaction = {
                'user_id': user_id,
                'product_id': product,
                'timestamp': time  # UŻYJ PODANEGO TIMESTAMP
            }
            self.transactions = pd.concat([self.transactions, pd.DataFrame([new_transaction])], ignore_index=True)
        # Dodaj do prostego modelu
        self.simple_model.add_transaction(user_id, products)
        
        # Przetrenuj zaawansowany model jeśli mamy wystarczająco danych
        self._needs_retraining = True
    
    def get_recommendations(self, user_id, current_cart):
        """Pobiera rekomendacje"""

        if self._needs_retraining and len(self.users) >= 3:
            self.advanced_model.fit(self.transactions, self.users, self.products)
            self._needs_retraining = False
        simple_recs = self.simple_model.recommend( current_cart)
        
        if user_id in self.advanced_model.user_vectors:
            advanced_recs = self.advanced_model.recommend(user_id, current_cart, self.transactions)
            # Połącz rekomendacje (możesz dodać ważenie)     
            all_recs = list(set(simple_recs + advanced_recs))
        else:
            all_recs = simple_recs
            
        return all_recs[:5]  # Zwróć max 5 rekomendacji
    
class ImprovedRecommendationModel:
    def __init__(self):
        self.user_vectors = {}
        self.product_categories = []
        self.sorted_user_ids = None 
        
    def prepare_features(self, transactions, users, products):
        """Przygotowuje cechy bez niepotrzebnego skalowania"""
        # Zbierz wszystkie kategorie produktów
        self.product_categories = products['category'].unique().tolist()
        
        user_vectors = {}
        
        for user_id in users['user_id']:
            user_data = users[users['user_id'] == user_id].iloc[0]
            user_transactions = transactions[transactions['user_id'] == user_id]
            
            vector = self.create_advanced_user_profile(
                user_id, 
                user_data['age'], 
                user_data['gender'],
                transactions,  # przekazujemy całe transactions
                self.product_categories,
                products
            )
            
            user_vectors[user_id] = vector
        
        return user_vectors
    
    def create_advanced_user_profile(self, user_id, age, gender, transactions, product_categories, products):
        """Tworzy zaawansowany profil użytkownika"""
        # 1. Cechy demograficzne (normalizujemy ręcznie)
        age_normalized = age / 100.0  # wiek w zakresie 0-1
        gender_encoded = 1 if gender == 'M' else 0
        
        demo_features = [age_normalized, gender_encoded]
        
        # 2. Preferencje kategorii
        user_transactions = transactions[transactions['user_id'] == user_id]
        user_products = user_transactions['product_id']
        
        category_preferences = []
        for category in product_categories:
            # Znajdź produkty w danej kategorii
            category_products = products[products['category'] == category]['product_id']
            # Policz ile razy user kupił produkty z tej kategorii
            category_count = user_products.isin(category_products).sum()
            total_user_products = len(user_products)
            preference = category_count / total_user_products if total_user_products > 0 else 0
            category_preferences.append(preference)
        
        
        user_vector = demo_features + category_preferences
        return np.array(user_vector)
    
    def fit(self, transactions, users, products):
        """Trenuje model na poprawnych cechach"""

        self.user_vectors = self.prepare_features(transactions, users, products)
        
        if len(self.user_vectors) > 1:
            self.sorted_user_ids = sorted(self.user_vectors.keys())
            vectors_array = np.array([self.user_vectors[uid] for uid in self.sorted_user_ids])
            self.nn_model = NearestNeighbors(
                n_neighbors=min(3, len(vectors_array)),
                metric='cosine'  # LEPSZE dla preferencji!
            )
            self.nn_model.fit(vectors_array)
    
    def recommend(self, user_id, current_cart, transactions_df, top_n=3):
        """Generuje rekomendacje z poprawnymi danymi"""
        if user_id not in self.user_vectors:
            return []
        
        user_vector = self.user_vectors[user_id].reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(user_vector)
        recommendations = {}
  
        
        for idx in indices[0]:
            similar_user_id = self.sorted_user_ids[idx]
            similar_user_transactions = transactions_df[transactions_df['user_id'] == similar_user_id]
            product_frequencies = similar_user_transactions['product_id'].value_counts(normalize=True)
            for product, frequency in product_frequencies.items():
                if product not in current_cart:
                # DODAJ częstotliwość zamiast 1
                    recommendations[product] = recommendations.get(product, 0) + frequency
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [product for product, count in sorted_recs[:top_n]]

class SimpleRecommendationModel:
    """Generates recommendations based on the cart. Looks what user have in cart and based on it
        suggests other product that is frequently bought together with this product
    """
    def __init__(self):
        self.co_occurrence = defaultdict(int)
        self.user_history = defaultdict(list)
        
    def add_transaction(self, user_id, products):
        self.user_history[user_id].extend(products)
        
        for product_a, product_b in combinations(set(products), 2):
            pair = tuple(sorted([product_a, product_b]))
            self.co_occurrence[pair] += 1
    
    def recommend(self, current_cart, top_n=3):

        recommendations = {}
        
        for product in current_cart:
            for (prod_a, prod_b), count in self.co_occurrence.items():
                if product == prod_a and prod_b not in current_cart:
                    recommendations[prod_b] = recommendations.get(prod_b, 0) + count
                elif product == prod_b and prod_a not in current_cart:
                    recommendations[prod_a] = recommendations.get(prod_a, 0) + count
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [product for product, score in sorted_recs[:top_n]]

products_data = {
            # Produkty podstawowe
            'mleko': 'nabiał', 'masło': 'nabiał', 'ser_żółty': 'nabiał', 'ser_biały': 'nabiał',
            'jogurt_owocowy': 'nabiał', 'jogurt_naturalny': 'nabiał', 'jogurt_grecki': 'nabiał',
            
            'chleb': 'piekarnia', 'bułki': 'piekarnia', 'chleb_razowy': 'piekarnia', 'bagietka': 'piekarnia',
            
            'jajka': 'spożywcze', 'mąka': 'spożywcze', 'cukier': 'spożywcze', 'olej': 'spożywcze',
            'ryż': 'spożywcze', 'makaron': 'spożywcze', 'płatki_śniadaniowe': 'spożywcze',
            'mąka_żytnia': 'spożywcze', 'cukier_wanilinowy': 'spożywcze', 'ryż_basmati': 'spożywcze',
            
            # Warzywa i owoce
            'pomidory': 'warzywa_owoce', 'ogórki': 'warzywa_owoce', 'jabłka': 'warzywa_owoce',
            'banany': 'warzywa_owoce', 'awokado': 'warzywa_owoce', 'szpinak': 'warzywa_owoce',
            'winogrona': 'warzywa_owoce', 'brokuły': 'warzywa_owoce', 'limonka': 'warzywa_owoce',
            
            # Mięsa i wędliny
            'wędlina_drobiowa': 'mięso_wędliny', 'parówki': 'mięso_wędliny', 'kielbasa': 'mięso_wędliny',
            'wędlina_wieprzowa': 'mięso_wędliny', 'szynka_surowa': 'mięso_wędliny', 'kurczak': 'mięso_wędliny',
            'łosoś': 'mięso_wędliny', 'burger_mrożony': 'mięso_wędliny',
            
            # Napoje
            'piwo': 'napoje', 'kawa': 'napoje', 'cola': 'napoje', 'energetyk': 'napoje',
            'herbata': 'napoje', 'wino_czerwone': 'napoje', 'wino_biale': 'napoje', 
            'sok_jabłkowy': 'napoje', 'herbata_owocowa': 'napoje', 'mleko_kokosowe': 'napoje',
            
            # Przekąski
            'chipsy': 'przekąski', 'orzeszki': 'przekąski', 'paluszki': 'przekąski',
            'słone_przekąski': 'przekąski', 'wafelki': 'przekąski', 'ciastka_maślane': 'przekąski',
            'lody_czekoladowe': 'przekąski',
            
            # Inne
            'pampersy': 'dzieci', 'chusteczki_nawilżane': 'dzieci',
            'pizza_mrożona': 'mrożonki', 'warzywa_mrożone': 'mrożonki', 'frytki': 'mrożonki',
            'sos_pomidorowy': 'sosy_przyprawy', 'oliwa': 'sosy_przyprawy', 'czosnek': 'sosy_przyprawy',
            'pieprz': 'sosy_przyprawy', 'curry': 'sosy_przyprawy', 'sos_sojowy': 'sosy_przyprawy',
            'imbir': 'sosy_przyprawy', 'keczup': 'sosy_przyprawy', 'majonez': 'sosy_przyprawy',
            'drożdże': 'piekarnia', 'czekolada': 'słodycze', 'orzechy': 'słodycze',
            'migdały': 'słodycze', 'miód': 'słodycze', 'dżem': 'słodycze', 'margaryna': 'tłuszcze',
            'oliwki': 'przekąski', 'siemie_lniane': 'zdrowa_żywność', 'płatki_owsiane': 'zdrowa_żywność'
        }
system = ShoppingRecommendationSystem(products_data)

users_data = [
    # Młode kobiety (rodziny z dziećmi) - starsze transakcje
    (1, ['mleko', 'chleb', 'masło', 'jogurt_owocowy', 'pampersy', 'banany'], 25, 'K', datetime.now() - timedelta(days=28)),
    (2, ['mleko', 'jajka', 'płatki_śniadaniowe', 'chleb', 'ser_żółty', 'pomidory'], 28, 'K', datetime.now() - timedelta(days=25)),
    (3, ['mleko', 'chleb', 'wędlina_drobiowa', 'ser_biały', 'ogórki', 'jabłka'], 32, 'K', datetime.now() - timedelta(days=22)),
    # Młodzi mężczyźni (studenci/single)
    (4, ['piwo', 'chipsy', 'kawa', 'makaron', 'sos_pomidorowy', 'parówki'], 22, 'M', datetime.now() - timedelta(days=20)),
    (5, ['energetyk', 'pizza_mrożona', 'cola', 'chleb', 'ser_żółty', 'kielbasa'], 24, 'M', datetime.now() - timedelta(days=18)),
    (6, ['ryż', 'kurczak', 'warzywa_mrożone', 'oliwa', 'czosnek', 'pieprz'], 26, 'M', datetime.now() - timedelta(days=16)),
    # Starsze osoby
    (7, ['herbata', 'bułki', 'wafelki', 'cukier', 'margaryna', 'dżem'], 68, 'K', datetime.now() - timedelta(days=15)),
    (8, ['kawa', 'chleb_razowy', 'wędlina_wieprzowa', 'ser_pleśniowy', 'oliwki', 'wino_czerwone'], 65, 'M', datetime.now() - timedelta(days=14)),
    # Rodziny
    (9, ['mleko', 'chleb', 'masło', 'jajka', 'ser_żółty', 'wędlina', 'pomidory', 'ogórki', 'jabłka', 'banany'], 35, 'K', datetime.now() - timedelta(days=12)),
    (10, ['mąka', 'cukier', 'jajka', 'mleko', 'olej', 'drożdże', 'czekolada', 'orzechy'], 38, 'M', datetime.now() - timedelta(days=10)),
    # Dodatkowe transakcje dla istniejących użytkowników (nowsze transakcje)
    (1, ['jajka', 'banany', 'pampersy', 'chusteczki_nawilżane'], 25, 'K', datetime.now() - timedelta(days=7)),
    (1, ['mleko', 'chleb', 'masło', 'jogurt_owocowy', 'sok_jabłkowy'], 25, 'K', datetime.now() - timedelta(days=3)),
    (4, ['piwo', 'orzeszki', 'paluszki', 'słone_przekąski'], 22, 'M', datetime.now() - timedelta(days=6)),
    (9, ['mleko', 'płatki_śniadaniowe', 'jogurt_naturalny', 'miód', 'migdały'], 35, 'K', datetime.now() - timedelta(days=5)),
    (6, ['ryż', 'łosoś', 'brokuły', 'sos_sojowy', 'imbir'], 26, 'M', datetime.now() - timedelta(days=4)),
    (2, ['mleko', 'jajka', 'chleb', 'awokado', 'szpinak', 'jogurt_grecki'], 28, 'K', datetime.now() - timedelta(days=3)),
    (8, ['ser_pleśniowy', 'wino_biale', 'winogrona', 'szynka_surowa', 'bagietka'], 65, 'M', datetime.now() - timedelta(days=2)),
    (3, ['kurczak', 'ryż_basmati', 'curry', 'mleko_kokosowe', 'limonka'], 32, 'K', datetime.now() - timedelta(days=2)),
    (5, ['burger_mrożony', 'frytki', 'keczup', 'majonez', 'cola', 'lody_czekoladowe'], 24, 'M', datetime.now() - timedelta(days=1)),
    (7, ['herbata_owocowa', 'ciastka_maślane', 'mleko', 'cukier_wanilinowy', 'jajka'], 68, 'K', datetime.now() - timedelta(days=1)),
    (10, ['mąka_żytnia', 'siemie_lniane', 'płatki_owsiane', 'miód', 'jogurt_naturalny'], 38, 'M', datetime.now()) 
    ]
# Dodawanie zakupów
for user_id, products, age, gender, time in users_data:
    system.add_purchase(user_id, products, age, gender,time)

# Pobieranie rekomendacji
current_cart = ['mleko','chleb']
recommendations = system.get_recommendations(1, current_cart)
print(f"Jeśli kupujesz {current_cart}, możesz też potrzebować: {recommendations}")

print("=== TEST 2: STUDENT ===")
current_cart_2 = ['piwo', 'chipsy']
recommendations_2 = system.get_recommendations(4, current_cart_2)
print(f"User 4 (M22): kupuje {current_cart_2}")
print(f"Rekomendacje: {recommendations_2}")
print()
print("=== TEST 3: OSOBA STARSZA ===")
current_cart_3 = ['herbata', 'bułki']
recommendations_3 = system.get_recommendations(7, current_cart_3)
print(f"User 7 (K68): kupuje {current_cart_3}")
print(f"Rekomendacje: {recommendations_3}")
print()

# Test 4: Osoba dbająca o zdrowie
print("=== TEST 4: ZDROWE ODŻYWIANIE ===")
current_cart_4 = ['ryż', 'kurczak']
recommendations_4 = system.get_recommendations(6, current_cart_4)
print(f"User 6 (M26): kupuje {current_cart_4}")
print(f"Rekomendacje: {recommendations_4}")
print()

# Test 5: Nowy użytkownik (cold start)
print("=== TEST 5: NOWY UŻYTKOWNIK ===")
current_cart_5 = ['mleko']
recommendations_5 = system.get_recommendations(11, current_cart_5)  # Nieistniejący user
print(f"User 11 (nowy): kupuje {current_cart_5}")
print(f"Rekomendacje (simple model): {recommendations_5}")