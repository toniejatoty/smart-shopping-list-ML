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

def stage3(stage1_result, stage2_candidates, final_k=10):

    new_candidates = [p for p in stage2_candidates]
    lines_new = "\n".join(
            f"  ID= {p['name']} | kategorie: {', '.join(p['categories'])}"
            for p in new_candidates
        )
    new_str = f"\nKandaci do uzupełnienia koszyka :\n{lines_new}\n"

    intent_desc = f"{stage1_result.get('reasoning', '')}. Kategorie: {', '.join(stage1_result.get('intent_categories', []))}"

    prompt = (
        "Jesteś rekomendatorem zakupów spożywczych.\n\n"
        f"Intencja zakupowa klienta:\n  {intent_desc}\n"
        f"{new_str}\n"
        f"Wybierz DOKŁADNIE {final_k} produktów.\n"
        "Zasady:\n"
        "1. PRZEPISZ wszystkie produkty z sekcji 'KUPIONE WCZEŚNIEJ' – klient regularnie je kupuje.\n"
        "2. Uzupełnij pozostałe miejsca produktami z sekcji 'Nowe produkty'.\n"
        "3. Używaj wyłącznie produktów z powyższych list.\n\n"
        "Odpowiedz WYŁĄCZNIE poprawnym JSON-em:\n"
        '{"recommended_ids": ["produkt1", "produkt2", ...]}'
    )

    time.sleep(RATE_LIMIT_SECONDS)  
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    # 5. Parsowanie wyniku
    try:
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        final_recommendations = json.loads(clean_text)
    except Exception as e:
        print(f"Błąd Stage 3: {e}")
        final_recommendations = {"recommended_ids": [p['id'] for p in stage2_candidates[:final_k]]}

    return final_recommendations

if __name__ == "__main__":
    test_data = {
        "session_id": "example-user-1",
        "history_baskets": [
            {
                "products": [
                    {"name": "Soda oczyszczona spożywcza bez antyzbrylaczy – 1000g", "id": "5902802802576"},
                    {"name": "Almond Drink unsweetened – Milsa", "id": "25026290"},
                    {"name": "Beef Jerky – Tarczyński", "id": "5908230536007"},
                    {"name": "Popcorn – 100 g", "id": "3577060000136"},
                    {"name": "G – Bake Rolls,7 Days – 150g", "id": "5201360655373"}
                ],
                "date": "0.0"
            },
            {
                "products": [
                    {"name": "Soda oczyszczona spożywcza bez antyzbrylaczy – 1000g", "id": "5902802802576"},
                    {"name": "Pistachio Delight – Carte d’Or", "id": "8711327672499"},
                    {"name": "Beef Jerky – Tarczyński", "id": "5908230536007"},
                    {"name": "Milk chocolate-covered banana – Biedronka – 180 g", "id": "5907554477454"},
                    {"name": "Popcorn – 100 g", "id": "3577060000136"},
                    {"name": "Tost pełnoziarnisty – Auchan – 500 g", "id": "5904215134251"}
                ],
                "date": "15.0"
            },
            {
                "products": [
                    {"name": "Soda oczyszczona spożywcza bez antyzbrylaczy – 1000g", "id": "5902802802576"},
                    {"name": "Beef Jerky – Tarczyński", "id": "5908230536007"},
                    {"name": "Pistachio Delight – Carte d’Or", "id": "8711327672499"},
                    {"name": "Ser włoszczowski – Auchan", "id": "2228326001442"},
                    {"name": "Almond Butter", "id": "5900617040053"}
                ],
                "date": "21.0"
            },
            {
                "products": [
                    {"name": "Soda oczyszczona spożywcza bez antyzbrylaczy – 1000g", "id": "5902802802576"},
                    {"name": "Beef Jerky – Tarczyński", "id": "5908230536007"},
                    {"name": "Pistachio Delight – Carte d’Or", "id": "8711327672499"},
                    {"name": "Ser włoszczowski – Auchan", "id": "2228326001442"},
                    {"name": "G – Bake Rolls,7 Days – 150g", "id": "5201360655373"}
                ],
                "date": "29.0"
            }
        ]
    }

    result = stage1(test_data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    result2 = stage2(result)
    print(json.dumps(result2, indent=2, ensure_ascii=False))
    result3 = stage3(result, result2, final_k=DEFAULT_FINAL_K)
    print(json.dumps(result3, indent=2, ensure_ascii=False))