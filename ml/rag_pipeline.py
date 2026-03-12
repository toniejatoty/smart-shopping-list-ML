# 1. cała historia zakopowa użytkownika i counter do tego każdego produktu co kiedyś kupił
# 2. biore n ostatnich koszyków również counter licze
# 3. tworze prompta który zawiera dni miedzy zakupami i koszyki oraz te powtarzające się produkt x powtarzalnosć (tutaj jedynie kontekst n ostanich koszyków)
# 4. dostaje w wyniku kategorie :......... produkty: ......... jakiś reasoning czemu to wziął:........
# 5. dla każdego produktu z OFF (14k) liczy jak bardzo wynik z gemini się pokrywa 
# 6a. dla każdej kategorii przewidzianej od gemini sprawdź czy wystepuje w produkcie z OFF jeżeli tak to score +3 oraz sprawdź czy wystepuje w nazwie jeżeli tak to +1.5
# 6b. dla każdego keyword przewidizanego przez llm sprawdź czy wystepuje w nazwie produktu OFF jeżeli tak to +2 i dla każdej kategori z produktu OFF sprawdź czy keyword od gemini jest w tych kategoriach jak jest to +1 score (jak już sprawdzi to break to znaczy max +1 może dać za kategorie)
# 7a. jeżeli produkt był wcześniej kupiony to +PREVIOUSLY_BOUGHT_BOOST
# 7b. ilośc kupionego w przeszłości * FREQ_BOOST_PER_BUY
# 7c. sumujemy cały score dla produktu
# 8. sortowanie wybiera DEFAULT_CANDIDATES (40 aktualnie)
# 9. cache scored to niby sprawdza ile z tych 40 predykcji był że wcześniej kiedykolwiek przez usera kupione
# 10. bierze intent(wynik z 1 etapu), kandydatów (wynik 2 etapu) i tworzy prompt (tutaj wazna uwaga jest duży nacisk na produkty które kiedyś były kupione można zmienic wagami ale to recall spada)
# 11. no i on zwraca wynik potem jakieś topk recall inne statystyki i w ogóle 




# Avg Recall@10:  0.5165  (51.65%)
# #RATE_LIMIT_SECONDS = 21.0
# DEFAULT_CANDIDATES = 40   # ile kandydatów pobieramy z cache w etapie 2
# DEFAULT_FINAL_K = 10      # ile produktów zwraca finałowy ranking
# PREVIOUSLY_BOUGHT_BOOST = 8.0   # bonus score za kupiony wcześniej produkt
# FREQ_BOOST_PER_BUY = 1.5        # dodatkowy bonus za każde kolejne kupienie
# MAX_CACHE_HITS_IN_PROMPT = 0   # max liczba cache-hitów w prompcie Stage 3 to są produkty które mają bardzo dużą wage i bardzo prawdopodobne że będą w wyniku bo już kiedyś użytkownik je kupił
# N_SAMPLES =5                   # na ilu danych ma testować
# _client: genai.Client | None = None

# Avg Recall@10:  0.0974  (9.74%)
# #RATE_LIMIT_SECONDS = 21.0
# DEFAULT_CANDIDATES = 40   # ile kandydatów pobieramy z cache w etapie 2
# DEFAULT_FINAL_K = 10      # ile produktów zwraca finałowy ranking
# PREVIOUSLY_BOUGHT_BOOST = 0.0   # bonus score za kupiony wcześniej produkt
# FREQ_BOOST_PER_BUY = 0.0        # dodatkowy bonus za każde kolejne kupienie
# MAX_CACHE_HITS_IN_PROMPT = 0   # max liczba cache-hitów w prompcie Stage 3 to są produkty które mają bardzo dużą wage i bardzo prawdopodobne że będą w wyniku bo już kiedyś użytkownik je kupił
# N_SAMPLES =5                   # na ilu danych ma testować
# _client: genai.Client | None = None

#   Avg Recall@10:  0.0752  (7.52%)
# RATE_LIMIT_SECONDS = 21.0
# DEFAULT_CANDIDATES = 100   # ile kandydatów pobieramy z cache w etapie 2
# DEFAULT_FINAL_K = 10      # ile produktów zwraca finałowy ranking
# PREVIOUSLY_BOUGHT_BOOST = 0   # bonus score za kupiony wcześniej produkt
# FREQ_BOOST_PER_BUY = 0        # dodatkowy bonus za każde kolejne kupienie
# MAX_CACHE_HITS_IN_PROMPT = 0   # max liczba cache-hitów w prompcie Stage 3 to są produkty które mają bardzo dużą wage i bardzo prawdopodobne że będą w wyniku bo już kiedyś użytkownik je kupił
# N_SAMPLES =5                   # na ilu danych ma testować


# Avg Recall@10:  0.0530  (5.30%)
# RATE_LIMIT_SECONDS = 21.0
# DEFAULT_CANDIDATES = 20   # ile kandydatów pobieramy z cache w etapie 2
# DEFAULT_FINAL_K = 10      # ile produktów zwraca finałowy ranking
# PREVIOUSLY_BOUGHT_BOOST = 0   # bonus score za kupiony wcześniej produkt
# FREQ_BOOST_PER_BUY = 0        # dodatkowy bonus za każde kolejne kupienie
# MAX_CACHE_HITS_IN_PROMPT = 0   # max liczba cache-hitów w prompcie Stage 3 to są produkty które mają bardzo dużą wage i bardzo prawdopodobne że będą w wyniku bo już kiedyś użytkownik je kupił
# N_SAMPLES =5                   # na ilu danych ma testować

from __future__ import annotations

import ast
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from dotenv import load_dotenv


project_root = Path(__file__).parent.parent
OFF_CSV = project_root / "prepare_data_for_model" / "OpenFoodData.csv"
KAGGLE_CSV = project_root / "ml" / "data" / "kaggle_prepared.csv"

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    print(f"[ENV] Wczytano .env z: {_env_path}")
else:
    print("[ENV] Brak pliku .env – używam zmiennej środowiskowej GEMINI_API_KEY")

GEMINI_MODEL = "gemma-3-27b-it"
RATE_LIMIT_SECONDS = 21.0
DEFAULT_CANDIDATES = 20   # ile kandydatów pobieramy z cache w etapie 2
DEFAULT_FINAL_K = 10      # ile produktów zwraca finałowy ranking
PREVIOUSLY_BOUGHT_BOOST = 0   # bonus score za kupiony wcześniej produkt
FREQ_BOOST_PER_BUY = 0        # dodatkowy bonus za każde kolejne kupienie
MAX_CACHE_HITS_IN_PROMPT = 0   # max liczba cache-hitów w prompcie Stage 3 to są produkty które mają bardzo dużą wage i bardzo prawdopodobne że będą w wyniku bo już kiedyś użytkownik je kupił
N_SAMPLES =5                   # na ilu danych ma testować
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client

_product_cache: dict[str, dict] = {}
_cache_loaded = False


def _parse_categories(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(c).strip().lower() for c in raw if c]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(c).strip().lower() for c in parsed if c]
    except Exception:
        pass
    return [raw.strip().lower()]


def load_product_cache(csv_path: Path = OFF_CSV) -> dict[str, dict]:

    global _product_cache, _cache_loaded
    if _cache_loaded:
        return _product_cache

    print(f"[CACHE] Wczytywanie cache z: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["Id", "Name", "Categories"])
    df["Id"] = df["Id"].astype(str)
    df = df.dropna(subset=["Name"])

    for _, row in df.iterrows():
        pid = row["Id"]
        _product_cache[pid] = {
            "name": str(row["Name"]).strip(),
            "categories": _parse_categories(row.get("Categories", "")),
        }

    _cache_loaded = True
    print(f"[CACHE] Załadowano {len(_product_cache):,} produktów")
    return _product_cache

def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Brak JSON w odpowiedzi: {text[:300]}")
    return json.loads(match.group())


def _call_gemini(prompt: str) -> str:
    print(f"[API] Wysyłam prompt ({len(prompt)} znaków)…")
    t0 = time.time()
    response = _get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
    elapsed = time.time() - t0
    text = response.text or ""
    print(f"[API] Odpowiedź: {elapsed:.1f}s  /  {len(text)} znaków")
    return text



def _build_intent_prompt(history_baskets: list[dict]) -> str:
    history_str = ""
    freq: Counter = Counter()
    for i, basket in enumerate(history_baskets, 1):
        days = basket.get("days", "?")
        prods = basket["products"] or []
        freq.update(prods)
        prods_str = ", ".join(prods) if prods else "(brak danych)"
        history_str += f"  Koszyk {i} (+{days} dni): {prods_str}\n"

    repeated = [p for p, cnt in freq.most_common(20) if cnt > 1]
    repeated_str = ""
    if repeated:
        lines = ", ".join(f"{p} (x{freq[p]})" for p in repeated)
        repeated_str = (
            f"\nProdukty kupowane WIELOKROTNIE (PRIORYTET – prawie pewne w następnym koszyku):\n"
            f"  {lines}\n"
        )

    return (
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


def analyze_intent(history_baskets: list[dict]) -> dict:
    prompt = _build_intent_prompt(history_baskets)
    raw = _call_gemini(prompt)
    try:
        intent = _extract_json(raw)
        print(f"[INTENT] {intent}")
        return intent
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[INTENT] Błąd parsowania odpowiedzi: {exc}")
        return {"categories": [], "keywords": [], "reasoning": str(exc)}

def _score_product(product: dict, categories: list[str], keywords: list[str]) -> float:

    score = 0.0
    name_lower = product["name"].lower()
    prod_cats = {c.lower() for c in product["categories"]}

    cats_lower = [c.lower() for c in categories]
    kws_lower = [k.lower() for k in keywords]

    for cat in cats_lower:
        if cat in prod_cats:
            score += 3.0
        if cat in name_lower:
            score += 1.5

    for kw in kws_lower:
        if kw in name_lower:
            score += 2.0
        for pc in prod_cats:
            if kw in pc:
                score += 1.0
                break  

    return score


def search_cache(
    intent: dict,
    previously_bought_ids: set[str],
    n_candidates: int = DEFAULT_CANDIDATES,
    purchase_counts: dict[str, int] | None = None,
) -> tuple[list[dict], list[str]]:
    cache = load_product_cache()
    categories = intent.get("categories", [])
    keywords = intent.get("keywords", [])
    purchase_counts = purchase_counts or {}

    scored: list[tuple[float, str]] = []   

    for pid, info in cache.items():
        base_score = _score_product(info, categories, keywords)
        freq = purchase_counts.get(pid, 1 if pid in previously_bought_ids else 0)

        history_bonus = (PREVIOUSLY_BOUGHT_BOOST + freq * FREQ_BOOST_PER_BUY
                         if pid in previously_bought_ids else 0.0)
        score = base_score + history_bonus

        if score <= 0:
            continue
        scored.append((score, pid))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_candidates]

    candidates = [
        {"id": pid, **cache[pid], "score": sc}
        for sc, pid in top
    ]

    cache_hit_ids = [p["id"] for p in candidates if p["id"] in previously_bought_ids]

    print(f"[CACHE] Scored {len(scored)} produktów(wynik większy od 0) (ilość unikalnych produktów używkonika :{len(previously_bought_ids)}) | "
          f"TOP {len(candidates)} kandydatów | cache_hits w TOP: {len(cache_hit_ids)}")

    return candidates, cache_hit_ids

def _build_ranking_prompt(
    intent: dict,
    candidates: list[dict],
    cache_hits_details: list[dict],
    final_k: int,
) -> str:
    intent_desc = (
        f"Kategorie: {', '.join(intent.get('categories', []))} | "
        f"Cechy: {', '.join(intent.get('keywords', []))} | "
    )

    cache_str = ""
    if cache_hits_details:
        lines = "\n".join(
            f"  {j}. ID={p['id']} | {p['name']}"
            for j, p in enumerate(cache_hits_details, 1)
        )
        cache_str = (
            f"\n⚑ PRODUKTY KUPIONE WCZEŚNIEJ ({len(cache_hits_details)} szt.) – "
            "WSTAW JE DO WYNIKÓW W PIERWSZEJ KOLEJNOŚCI:\n" + lines + "\n"
        )

    cache_hit_id_set = {p["id"] for p in cache_hits_details}
    new_candidates = [p for p in candidates if p["id"] not in cache_hit_id_set]
    new_str = ""
    if new_candidates:
        lines_new = "\n".join(
            f"  ID={p['id']} | {p['name']} | kategorie: {', '.join(p['categories'])}"
            for p in new_candidates
        )
        new_str = f"\nNowe produkty do uzupełnienia koszyka:\n{lines_new}\n"

    return (
        "Jesteś rekomendatorem zakupów spożywczych.\n\n"
        f"Intencja zakupowa klienta:\n  {intent_desc}\n"
        f"{cache_str}"
        f"{new_str}\n"
        f"Wybierz DOKŁADNIE {final_k} produktów.\n"
        "Zasady:\n"
        "1. PRZEPISZ wszystkie produkty z sekcji 'KUPIONE WCZEŚNIEJ' – klient regularnie je kupuje.\n"
        "2. Uzupełnij pozostałe miejsca produktami z sekcji 'Nowe produkty'.\n"
        "3. Używaj wyłącznie ID z powyższych list.\n\n"
        "Odpowiedz WYŁĄCZNIE poprawnym JSON-em:\n"
        '{"recommended_ids": ["id1", "id2", ...]}'
    )


def rank_candidates(
    intent: dict,
    candidates: list[dict],
    cache_hits: list[str],
    final_k: int = DEFAULT_FINAL_K,
) -> list[dict]:
    cache = load_product_cache()

    cache_hits_details = [
        {"id": pid, **cache[pid]}
        for pid in cache_hits[:MAX_CACHE_HITS_IN_PROMPT]
        if pid in cache
    ]

    all_for_ranking = {p["id"]: p for p in candidates}
    all_for_ranking.update({p["id"]: p for p in cache_hits_details})
    all_for_ranking_list = list(all_for_ranking.values())

    if not all_for_ranking_list:
        print("[RANK] Brak kandydatów do rankingu")
        return []

    time.sleep(RATE_LIMIT_SECONDS) 
    prompt = _build_ranking_prompt(intent, candidates, cache_hits_details, final_k)
    raw = _call_gemini(prompt)

    try:
        result = _extract_json(raw)
        recommended_ids: list[str] = result.get("recommended_ids", [])
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[RANK] Błąd parsowania: {exc} – fallback na TOP {final_k} z cache_hits+kandydatów")
        recommended_ids = [p["id"] for p in cache_hits_details[:final_k]]
        if len(recommended_ids) < final_k:
            existing = set(recommended_ids)
            for p in candidates:
                if p["id"] not in existing:
                    recommended_ids.append(p["id"])
                if len(recommended_ids) >= final_k:
                    break

    final = [all_for_ranking[rid] for rid in recommended_ids if rid in all_for_ranking]

    if len(final) < final_k:
        existing_ids = {p["id"] for p in final}
        for p in cache_hits_details + candidates:
            if p["id"] not in existing_ids:
                final.append(p)
                existing_ids.add(p["id"])
            if len(final) >= final_k:
                break

    print(f"[RANK] Finalnych rekomendacji: {len(final)}")
    return final[:final_k]

def run_rag_pipeline(
    history_baskets: list[dict],
    previously_bought_ids: set[str] | None = None,
    purchase_counts: dict[str, int] | None = None,
    n_candidates: int = DEFAULT_CANDIDATES,
    final_k: int = DEFAULT_FINAL_K,
) -> dict[str, Any]:
    previously_bought_ids = previously_bought_ids or set()

    print("\n" + "=" * 60)
    print("RAG-LEJEK: START")
    print("=" * 60)

    print("\n[ETAP 1] Analiza intencji zakupowej…")
    intent = analyze_intent(history_baskets)

    print("\n[ETAP 2] Wyszukiwanie kandydatów w cache…")
    candidates, cache_hits = search_cache(
        intent, previously_bought_ids, n_candidates, purchase_counts
    )

    print("\n[ETAP 3] Finalny ranking przez Gemini…")
    final = rank_candidates(intent, candidates, cache_hits, final_k)

    output = {
        "db_search_instruction": intent,       
        "cache_hits": cache_hits,              
        "final_recommendations": final,        
    }

    print("\n" + "=" * 60)
    print("RAG-LEJEK: KONIEC")
    print(f"  Intencja:       {intent.get('reasoning', '')}")
    print(f"  Cache hits:     {len(cache_hits)}")
    print(f"  Rekomendacje:   {len(final)}")
    print("=" * 60)
    return output


def _recall_at_k_ids(predicted_ids: list[str], true_ids: list[str], k: int) -> float:
    if not true_ids:
        return 0.0
    top_pred = set(predicted_ids[:k])
    hits = sum(1 for tid in true_ids if tid in top_pred)
    return hits / len(true_ids)


def evaluate_rag_pipeline(
    n_samples: int = 10,
    history_baskets: int = 8,
    final_k: int = 10,
    n_candidates: int = DEFAULT_CANDIDATES,
) -> dict:
    load_product_cache(OFF_CSV)
    id2name: dict[str, str] = {pid: info["name"] for pid, info in _product_cache.items()}

    print(f"\n[EVAL] Wczytywanie {KAGGLE_CSV.name}…")
    df = pd.read_csv(KAGGLE_CSV)
    df["off_product_id"] = df["off_product_id"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    min_orders = history_baskets + 1
    valid_users = [uid for uid, grp in df.groupby("user_id") if len(grp) >= min_orders]
    if not valid_users:
        raise RuntimeError(f"Brak użytkowników z >= {min_orders} zamówieniami")

    if len(valid_users) < n_samples:
        n_samples = len(valid_users)
        print(f"[EVAL] Zmniejszam n_samples do {n_samples}")

    step = max(1, len(valid_users) // n_samples)
    chosen = valid_users[::step][:n_samples]
    print(f"[EVAL] Wybrano {len(chosen)} użytkowników do ewaluacji  (co {step}-ty z {len(valid_users)})")

    results = []
    recalls = []
    total_start = time.time()

    for i, uid in enumerate(chosen, 1):
        print(f"\n{'─'*55}")
        print(f"[EVAL {i}/{len(chosen)}] user_id={uid}")

        user_df = df[df["user_id"] == uid].sort_values("order_id").reset_index(drop=True)

        history_rows = user_df.iloc[:-1]
        purchase_counter: Counter = Counter()
        for ids in history_rows["off_product_id"]:
            purchase_counter.update(str(x) for x in ids)
        all_bought: set[str] = set(purchase_counter.keys())

        context_rows = history_rows.iloc[-history_baskets:]
        target_row = user_df.iloc[-1]
        target_ids = [str(x) for x in target_row["off_product_id"] if str(x) in _product_cache]
        target_names = [id2name[tid] for tid in target_ids]

        print(f"[EVAL] Target ({len(target_ids)} prod.): {target_names}")

        baskets = []
        for _, row in context_rows.iterrows():
            names = [id2name[str(x)] for x in row["off_product_id"] if str(x) in id2name]
            days = int(row["days_since_prior_order"]) if pd.notna(row.get("days_since_prior_order")) else 0
            baskets.append({"days": days, "products": names})

        try:
            output = run_rag_pipeline(
                history_baskets=baskets,
                previously_bought_ids=all_bought,
                purchase_counts=dict(purchase_counter),
                n_candidates=n_candidates,
                final_k=final_k,
            )
            predicted_ids = [p["id"] for p in output["final_recommendations"]]
            recall = _recall_at_k_ids(predicted_ids, target_ids, k=final_k)
        except Exception as exc:
            print(f"[EVAL] Błąd pipeline: {exc}")
            predicted_ids = []
            recall = 0.0

        predicted_names = [
            _product_cache[pid]["name"] for pid in predicted_ids if pid in _product_cache
        ]

        recalls.append(recall)
        results.append({
            "user_id": uid,
            "target_ids": target_ids,
            "target_names": target_names,
            "predicted_ids": predicted_ids,
            f"recall@{final_k}": round(recall, 4),
        })

        print(
            f"\n[EVAL] Koszyk docelowy ({len(target_names)} prod.):\n"
            + "\n".join(f"        - {n}" for n in target_names)
            + f"\n\n[EVAL] Przewidział ({len(predicted_names)} prod.):\n"
            + "\n".join(f"        - {n}" for n in predicted_names)
            + f"\n\n[EVAL] Recall@{final_k} = {recall:.4f}  ({recall*100:.1f}%)"
        )

        if i < len(chosen):
            time.sleep(RATE_LIMIT_SECONDS)

    elapsed = time.time() - total_start
    avg = sum(recalls) / len(recalls) if recalls else 0.0

    print("\n" + "=" * 55)
    print("WYNIKI EWALUACJI RAG-LEJEK")
    print("=" * 55)
    print(f"  Próbki:          {len(recalls)}")
    print(f"  Top-K:           {final_k}")
    print(f"  Avg Recall@{final_k}:  {avg:.4f}  ({avg * 100:.2f}%)")
    print(f"  Min Recall@{final_k}:  {min(recalls):.4f}")
    print(f"  Max Recall@{final_k}:  {max(recalls):.4f}")
    print(f"  Czas:            {elapsed:.1f}s")
    print("=" * 55)

    return {
        "avg_recall": round(avg, 4),
        "min_recall": round(min(recalls), 4),
        "max_recall": round(max(recalls), 4),
        "n_samples": len(recalls),
        "top_k": final_k,
        "elapsed_s": round(elapsed, 1),
        "details": results,
    }


if __name__ == "__main__":
    stats = evaluate_rag_pipeline(n_samples=N_SAMPLES, history_baskets=8, final_k=10)
    print("\n────── SUMMARY JSON ──────")
    print(json.dumps({k: v for k, v in stats.items() if k != "details"},
                     ensure_ascii=False, indent=2))
