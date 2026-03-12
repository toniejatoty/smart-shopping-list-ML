from google import genai
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import ast
import re
import pandas as pd
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

_env_paths = Path(__file__).parent / ".env"

if _env_paths.exists():
    load_dotenv(_env_paths)
    print(f"[ENV] Wczytano .env z: {_env_paths}")
else:
    print("[ENV] Brak pliku .env – używam zmiennej środowiskowej GEMINI_API_KEY")

client = genai.Client()  
GEMINI_MODEL = "gemma-3-27b-it"

RATE_LIMIT_SECONDS = 2.0

project_root = Path(__file__).parent.parent
KAGGLE_CSV = project_root / "ml" / "data" / "kaggle_prepared.csv"
OFF_CSV    = project_root / "prepare_data_for_model" / "OpenFoodData.csv"
PRODUCT_CATEGORY_MODEL_PATH = project_root.parent / "product_category_model"


def load_id_to_name(off_csv_path):
    print(f"[DATA] Wczytywanie OpenFoodData z: {off_csv_path}")
    print(f"[DATA] Plik istnieje: {off_csv_path.exists()}")
    df = pd.read_csv(off_csv_path, usecols=["Id", "Name"])
    df["Id"] = df["Id"].astype(str)
    df = df.dropna(subset=["Name"])
    mapping = dict(zip(df["Id"], df["Name"].str.strip()))
    print(f"[DATA] Załadowano {len(mapping):,} nazw produktów")
    sample = list(mapping.items())[:3]
    print(f"[DATA] Przykłady: {sample}")
    return mapping

def ids_to_names(id_list, id2name):
    names = [id2name[str(i)] for i in id_list if str(i) in id2name]
    return names


def parse_gemini_product_list(text):
    text = re.sub(r"\*\*.*?\*\*", "", text)
    text = re.sub(r"#{1,4}\s.*", "", text)
    parts = re.split(r"[\n,]+", text)
    products = []
    for part in parts:
        clean = re.sub(r"^[\s\d\.\-\*\•]+", "", part).strip()
        if len(clean) > 2:
            products.append(clean.lower())
    return products


def recall_at_k(predicted, true_items, k, fuzzy_threshold=1):
    pred_top = [p.lower() for p in predicted[:k]]
    true_list = [t.lower() for t in true_items]
    if not true_list:
        return 0.0, []
    hits = 0
    matched = []
    for true in true_list:
        best_score, best_pred = 0.0, None
        for pred in pred_top:
            score = SequenceMatcher(None, pred, true).ratio()
            if score > best_score:
                best_score, best_pred = score, pred
        if best_score >= fuzzy_threshold:
            hits += 1
            matched.append((best_pred, true, f"{best_score:.2f}"))
    return hits / len(true_list), matched


def ask_gemini_with_retry(prompt):
    print(f"[API]  Wysyłam zapytanie (długość promptu: {len(prompt)} znaków)")
    t0 = time.time()

    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt
    )
    elapsed = time.time() - t0
    text = response.text if response.text else ""
    print(f"[API]  Odpowiedź odebrana w {elapsed:.1f}s  ({len(text)} znaków)")
    return text

def evaluate_gemini_recommender(
    num_users = 10,
    history_baskets = 8,
    top_k = 10
):
    print("\n" + "=" * 60)
    print("EWALUACJA GEMINI JAKO REKOMENDATORA ZAKUPÓW")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Użytkownicy: {num_users}  |  Historia: {history_baskets} koszyków  |  Top-K: {top_k}")
    print("=" * 60)

    id2name = load_id_to_name(OFF_CSV)
    print(f"\n[DATA] Wczytywanie {KAGGLE_CSV.name} ...")
    print(f"[DATA] Plik istnieje: {KAGGLE_CSV.exists()}")
    df = pd.read_csv(KAGGLE_CSV)
    print(f"[DATA] Surowe wiersze: {len(df):,}  |  kolumny: {df.columns.tolist()}")
    df["off_product_id"] = df["off_product_id"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    print(f"[DATA] Unikalni użytkownicy: {df['user_id'].nunique():,}")

    min_orders = history_baskets + 1
    valid_users = [
        uid for uid, grp in df.groupby("user_id") if len(grp) >= min_orders
    ]
    print(f"[DATA] Użytkownicy z >={min_orders} zamówieniami: {len(valid_users):,}")

    if len(valid_users) < num_users:
        num_users = len(valid_users)
        print(f"[DATA] Zmniejszam num_users do {num_users}")

    step = max(1, len(valid_users) // num_users)
    selected_users = valid_users[::step][:num_users]
    print(f"[DATA] Wybranych użytkowników: {len(selected_users)}  (co {step}-ty)")

    recalls = []
    total_start = time.time()

    for i, uid in enumerate(selected_users, 1):
        print(f"\n{'─'*50}")
        print(f"[USER {i}/{num_users}] user_id={uid}")

        user_orders = (
            df[df["user_id"] == uid]
            .sort_values("order_id")
            .reset_index(drop=True)
        )
        print(f"[USER] Łącznie zamówień: {len(user_orders)}  (używam wszystkich jako historia)")

        all_context_rows = user_orders.iloc[:-1]
        context_rows = all_context_rows.iloc[-history_baskets:]
        target_row   = user_orders.iloc[-1]

        history_names = [
            ids_to_names(row["off_product_id"], id2name)
            for _, row in context_rows.iterrows()
        ]
        target_names = ids_to_names(target_row["off_product_id"], id2name)

        print(f"[USER] Wybranych koszyków do promptu: {len(context_rows)} / {len(all_context_rows)} (cała historia: freq z {len(all_context_rows)})")
        print(f"[USER] Koszyk docelowy ({len(target_names)} prod.): {target_names}")

        all_history_names_full = [
            ids_to_names(row["off_product_id"], id2name)
            for _, row in all_context_rows.iterrows()
        ]
        all_history_products = [p for basket in all_history_names_full for p in basket]
        freq = Counter(all_history_products)
        repeated = [p for p, cnt in freq.most_common() if cnt > 1]
        print(f"[USER] Powtarzające się produkty ({len(repeated)}): {repeated[:5]}")

        history_str = ""
        for j, (_, row) in enumerate(context_rows.iterrows(), 1):
            basket = history_names[j - 1]
            basket_str = ", ".join(basket) if basket else "(brak danych)"
            days = int(row["days_since_prior_order"]) if pd.notna(row.get("days_since_prior_order")) else "?"
            history_str += f"  Koszyk {j} (+{days} dni): {basket_str}\n"

        repeated_str = ""
        if repeated:
            top_repeated = [(p, freq[p]) for p in repeated[:40]]
            lines = ", ".join(f"{p} (x{cnt})" for p, cnt in top_repeated)
            repeated_str = f"\nProdukty kupowane wielokrotnie (posortowane od najczęstszych):\n{lines}\n"

        prompt = (
            "Jesteś analitykiem zakupów. Przewidz następny koszyk zakupowy klienta.\n\n"
            f"Historia {len(history_names)} kolejnych koszyków (od najstarszego, liczby = dni między zakupami):\n{history_str}\n"
            f"{repeated_str}\n"
            "Zasady (KRYTYCZNE):\n"
            "1. PRZEPISZ produkty z listy 'kupowane wielokrotnie' – są prawie pewne.\n"
            "2. Uzupełnij pozostałe miejsca produktami z historii.\n"
            "3. Używaj IDENTYCZNYCH nazw jak w historii – nie skracaj, nie tłumacz.\n\n"
            f"Podaj dokładnie {top_k} nazw produktów, każdą w osobnej linii.\n"
            "Bez numerów, bez opisu, bez dodatkowego tekstu."
        )
        
        response_text = ask_gemini_with_retry(prompt)

        predicted = parse_gemini_product_list(response_text)
        r, matched = recall_at_k(predicted, target_names, k=top_k)
        recalls.append(r)

        print(f"[WYNIK] Predykcje ({len(predicted)}, używam top {top_k}): {predicted[:top_k]}")
        print(f"[WYNIK] Prawdziwy koszyk ({len(target_names)}): {target_names}")
        print(f"[WYNIK] Dopasowania (fuzzy): {matched}")
        print(f"[WYNIK] Recall@{top_k} = {r:.4f}  ({r*100:.1f}%)")

        if i < num_users:
            print(f"[API]  Czekam {RATE_LIMIT_SECONDS}s (rate limit)...")
            time.sleep(RATE_LIMIT_SECONDS)

    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("WYNIKI KOŃCOWE")
    print("=" * 60)
    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        print(f"Liczba próbek:   {len(recalls)}")
        print(f"Top-K:           {top_k}")
        print(f"Avg Recall@{top_k}: {avg_recall:.4f}  ({avg_recall*100:.2f}%)")
        print(f"Min Recall@{top_k}: {min(recalls):.4f}")
        print(f"Max Recall@{top_k}: {max(recalls):.4f}")
        print(f"Czas całkowity:  {elapsed:.1f}s")
    else:
        print("Brak wyników – wszystkie próbki pominięte.")
    print("=" * 60)
    return recalls

if __name__ == "__main__":
    evaluate_gemini_recommender(num_users=5, history_baskets=8, top_k=10)
