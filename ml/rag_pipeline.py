import json
import os
from collections import Counter
from google import genai
from dotenv import load_dotenv
import pandas as pd 
import time
load_dotenv()

GEMINI_MODEL = "gemma-3-27b-it"
RATE_LIMIT_SECONDS = 21.0
DEFAULT_CANDIDATES = 40   # ile kandydatów pobieramy z cache w etapie 2
DEFAULT_FINAL_K = 10      # ile produktów zwraca finałowy ranking
PREVIOUSLY_BOUGHT_BOOST = 8   # bonus score za kupiony wcześniej produkt
FREQ_BOOST_PER_BUY = 1.5        # dodatkowy bonus za każde kolejne kupienie   

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def stage1(json_data):
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    history_baskets = json_data.get("history_baskets", [])
    
    purchase_counts = Counter()
    user_bought_ids = []
    history_lines = []
    for i, basket in enumerate(history_baskets):
        products = basket.get("products", [])
        days = basket.get("date", "0")
        
        prod_names = []
        for p in products:
            p_id = p['id']
            p_name = p['name'][0]
            
            purchase_counts[p_name] += 1
            if p_id not in user_bought_ids:
                user_bought_ids.append(p_id)
            
            prod_names.append(p_name)
        
        history_lines.append(f"  Koszyk {i+1} (+{days} dni): {', '.join(prod_names)}")

    history_str = "\n".join(history_lines)

    repeated_list = [
        f"- {pid} (kupiono {count} razy)" 
        for pid, count in purchase_counts.items() if count > 1
    ]
    
    repeated_str = "\nProdukty kupowane regularnie:\n" + "\n".join(repeated_list)
    
    prompt = (
        "Jesteś analitykiem zakupów spożywczych.\n\n"
        "Masz historię zakupów klienta (od najstarszego do najnowszego):\n"
        f"{history_str}"
        f"{repeated_str}\n"
        "Na podstawie tej historii określ, CZEGO klient potrzebuje TERAZ "
        "(uwzględnij cykliczność zakupów i uzupełnianie koszyka).\n\n"
        "Odpowiedz WYŁĄCZNIE poprawnym JSON-em (bez komentarzy, bez markdown):\n"
        "{\n"
        '  "categories": ["kategoria1", "kategoria2"],\n'
        '  "keywords": ["dokładna_nazwa_produktu1", "slowo_klucz2"],\n'
        '  "reasoning": "max 2 zdania"\n'
        "}\n"
        "Wpisuj w keywords DOKŁADNE nazwy często kupowanych produktów z historii."
    )
    time.sleep(RATE_LIMIT_SECONDS)  
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    try:
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        llm_output = json.loads(clean_text)
    except Exception as e:
        print(f"Błąd parsowania JSON: {e}")
        llm_output = {"categories": [], "keywords": [], "reasoning": "error"}

    final_output = {
        "intent_categories": llm_output.get("categories", []),
        "intent_keywords": llm_output.get("keywords", []),
        "user_bought_ids": user_bought_ids,
        "purchase_counts": dict(purchase_counts)
    }

    with open(os.path.join("cache", "stage1_output.json"),'w') as f:
        json.dump(final_output, f)
        f.write("\n")
    return final_output


def stage2(stage1_result, csv_path=r'C:\Users\konra\Desktop\vscode\projekt_zespolowy\smart-shopping-list-ML\prepare_data_for_model\OpenFoodData.csv'):
    df = pd.read_csv(csv_path)

    def parse_categories(x):
        try:
            return [cat.lower() for cat in ast.literal_eval(x)]
        except:
            return []

    df['parsed_categories'] = df['Categories'].apply(parse_categories)
    df['name_lower'] = df['Name'].fillna('').str.lower()
    
    intent_categories = [c.lower() for c in stage1_result.get("intent_categories", [])]
    intent_keywords = [k.lower() for k in stage1_result.get("intent_keywords", [])]
    user_bought_ids = stage1_result.get("user_bought_ids", [])
    purchase_counts = stage1_result.get("purchase_counts", {})

    scores = []

    for _, row in df.iterrows():
        score = 0.0
        p_id = str(row['Id'])
        p_name = row['name_lower']
        p_cats = row['parsed_categories']

        for cat in intent_categories:
            if cat in p_cats:
                score += 3.0
            if cat in p_name:
                score += 1.5

        for kw in intent_keywords:
            if kw in p_name:
                score += 2.0
            
            for p_cat in p_cats:
                if kw in p_cat:
                    score += 1.0
                    break 

        if p_id in user_bought_ids:
            score += PREVIOUSLY_BOUGHT_BOOST
            count = purchase_counts.get(p_id, 0)
            score += (count * FREQ_BOOST_PER_BUY)

        if score > 0:
            scores.append({
                "id": p_id,
                "name": row['Name'],
                "score": score,
                "categories": row['Categories']
            })

    ranked_products = sorted(scores, key=lambda x: x['score'], reverse=True)
    return ranked_products[:DEFAULT_CANDIDATES]

def stage3(stage2_candidates, final_k=10):
    with open(os.path.join("cache", "stage1_output.json"), 'r') as f:
        stage1_result = json.load(f)
    counts = Counter(stage1_result['purchase_counts'])
    top_frequent_names = [name for name, count in counts.most_common(15)]
    
    cache_lines = [f"- {name}" for name in top_frequent_names]
    cache_str = "\n".join(cache_lines)

    new_candidates = [p for p in stage2_candidates if p["name"] not in top_frequent_names]
    
    new_lines = [f"- {p['name']} (Kategorie: {p['categories']})" for p in new_candidates[:20]]
    new_str = "\n".join(new_lines)

    prompt = (
        "Jesteś inteligentnym asystentem zakupowym.\n\n"
        "PRODUKTY KUPOWANE REGULARNIE (Priorytet):\n"
        f"{cache_str}\n\n"
        "NOWE PROPOZYCJE (Uzupełnienie):\n"
        f"{new_str}\n\n"
        f"ZADANIE: Wybierz dokładnie {final_k} produktów do nowego koszyka.\n"
        "ZASADY:\n"
        "1. Wybierz najpierw produkty z listy REGULARNYCH, które pasują do aktualnych potrzeb.\n"
        "2. Jeśli zostanie miejsce, dobierz najlepsze NOWE PROPOZYCJE.\n"
        "3. Zwróć wyłącznie JSON z listą nazw.\n\n"
        '{"recommended_products": ["nazwa1", "nazwa2"]}'
    )

    time.sleep(RATE_LIMIT_SECONDS)  
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    try:
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        final_recommendations = json.loads(clean_text)
    except Exception as e:
        print(f"Błąd Stage 3: {e}")
        final_recommendations = {"recommended_ids": [p['id'] for p in stage2_candidates[:final_k]]}

    return final_recommendations

