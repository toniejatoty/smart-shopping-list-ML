from datetime import datetime, timedelta
import json
from pathlib import Path
users_data = [

    # User 1 - Młoda kobieta (rodzina z dziećmi)
    (1, ['mleko', 'chleb', 'masło', 'jogurt_owocowy', 'pampersy', 'banany'], 25, 'K', datetime.now() - timedelta(days=14)),
    (1, ['jajka', 'banany', 'pampersy', 'chusteczki_nawilżane', 'mydło'], 25, 'K', datetime.now() - timedelta(days=12)),
    (1, ['mleko', 'chleb', 'masło', 'jogurt_owocowy', 'sok_jabłkowy', 'biszkopty'], 25, 'K', datetime.now() - timedelta(days=10)),
    (1, ['ser_żółty', 'wędlina_drobiowa', 'pomidory', 'ogórki', 'jabłka', 'herbatniki'], 25, 'K', datetime.now() - timedelta(days=7)),
    (1, ['mleko', 'płatki_śniadaniowe', 'kakao', 'jogurt_naturalny', 'gruszki', 'chusteczki'], 25, 'K', datetime.now() - timedelta(days=3)),
    (1, ['mąka', 'cukier', 'jajka', 'olej', 'drożdże', 'rodzynki', 'wanilia'], 25, 'K', datetime.now() - timedelta(days=1)),

    # User 2 - Młoda kobieta
    (2, ['mleko', 'jajka', 'płatki_śniadaniowe', 'chleb', 'ser_żółty', 'pomidory'], 28, 'K', datetime.now() - timedelta(days=13)),
    (2, ['mleko', 'jajka', 'chleb', 'awokado', 'szpinak', 'jogurt_grecki'], 28, 'K', datetime.now() - timedelta(days=11)),
    (2, ['ryż', 'kurczak', 'brokuły', 'marchewka', 'cebula', 'czosnek'], 28, 'K', datetime.now() - timedelta(days=9)),
    (2, ['makaron', 'sos_pomidorowy', 'ser_mozzarella', 'bazylia', 'oliwa', 'pomidory'], 28, 'K', datetime.now() - timedelta(days=6)),
    (2, ['mleko_migdałowe', 'płatki_owsiane', 'orzechy', 'miód', 'jagody', 'siemie_lniane'], 28, 'K', datetime.now() - timedelta(days=2)),

    # User 3 - Kobieta (rodzina)
    (3, ['mleko', 'chleb', 'wędlina_drobiowa', 'ser_biały', 'ogórki', 'jabłka'], 32, 'K', datetime.now() - timedelta(days=15)),
    (3, ['kurczak', 'ryż_basmati', 'curry', 'mleko_kokosowe', 'limonka', 'kolendra'], 32, 'K', datetime.now() - timedelta(days=12)),
    (3, ['ziemniaki', 'marchew', 'pietruszka', 'seler', 'wołowina', 'liść_laurowy'], 32, 'K', datetime.now() - timedelta(days=9)),
    (3, ['łosoś', 'szparagi', 'cytryna', 'koperek', 'ziemniaki_młode', 'koper'], 32, 'K', datetime.now() - timedelta(days=6)),
    (3, ['jogurt_dla_dzieci', 'chrupki_kukurydziane', 'serek_homogenizowany', 'mus_owocowy', 'biszkopty'], 32, 'K', datetime.now() - timedelta(days=2)),

    # User 4 - Młody mężczyzna (student)
    (4, ['piwo', 'chipsy', 'kawa', 'makaron', 'sos_pomidorowy', 'parówki'], 22, 'M', datetime.now() - timedelta(days=14)),
    (4, ['piwo', 'orzeszki', 'paluszki', 'słone_przekąski', 'krakersy'], 22, 'M', datetime.now() - timedelta(days=11)),
    (4, ['energetyk', 'czekolada', 'bułki', 'ser_topiony', 'szynka', 'musztarda'], 22, 'M', datetime.now() - timedelta(days=8)),
    (4, ['mrożona_pizza', 'frytki', 'keczup', 'majonez', 'cola', 'lody'], 22, 'M', datetime.now() - timedelta(days=5)),
    (4, ['ryż_jaśminowy', 'warzywa_mrożone', 'sos_sojowy', 'imbir', 'czosnek', 'olej_sezamowy'], 22, 'M', datetime.now() - timedelta(days=1)),

    # User 5 - Młody mężczyzna
    (5, ['energetyk', 'pizza_mrożona', 'cola', 'chleb', 'ser_żółty', 'kielbasa'], 24, 'M', datetime.now() - timedelta(days=13)),
    (5, ['burger_mrożony', 'frytki', 'keczup', 'majonez', 'cola', 'lody_czekoladowe'], 24, 'M', datetime.now() - timedelta(days=10)),
    (5, ['piwo_craft', 'ser_camembert', 'oliwki', 'bagietka', 'winogrona', 'szynka'], 24, 'M', datetime.now() - timedelta(days=7)),
    (5, ['makaron_penne', 'boczek', 'śmietana', 'pieczarki', 'cebula', 'parmezan'], 24, 'M', datetime.now() - timedelta(days=4)),
    (5, ['jajka', 'bekon', 'tosty', 'awokado', 'pomidory', 'sok_pomarańczowy'], 24, 'M', datetime.now() - timedelta(days=1)),

    # User 6 - Mężczyzna (single)
    (6, ['ryż', 'kurczak', 'warzywa_mrożone', 'oliwa', 'czosnek', 'pieprz'], 26, 'M', datetime.now() - timedelta(days=12)),
    (6, ['ryż', 'łosoś', 'brokuły', 'sos_sojowy', 'imbir', 'cytryna'], 26, 'M', datetime.now() - timedelta(days=10)),
    (6, ['wołowina', 'ziemniaki', 'cebula', 'czosnek', 'papryka', 'przyprawy'], 26, 'M', datetime.now() - timedelta(days=8)),
    (6, ['tuńczyk', 'makaron', 'kukurydza', 'groszek', 'majonez', 'jogurt'], 26, 'M', datetime.now() - timedelta(days=5)),
    (6, ['jajka', 'szynka', 'ser', 'pomidory', 'pieczarki', 'masło'], 26, 'M', datetime.now() - timedelta(days=2)),

    # User 7 - Starsza kobieta
    (7, ['herbata', 'bułki', 'wafelki', 'cukier', 'margaryna', 'dżem'], 68, 'K', datetime.now() - timedelta(days=14)),
    (7, ['herbata_owocowa', 'ciastka_maślane', 'mleko', 'cukier_wanilinowy', 'jajka'], 68, 'K', datetime.now() - timedelta(days=11)),
    (7, ['kawa_ziarnista', 'mleko_3,2%', 'bułki_maślane', 'miód', 'masło_ekstra'], 68, 'K', datetime.now() - timedelta(days=8)),
    (7, ['ryż_okrągły', 'mleko_do_gotowania', 'cukier_wanilia', 'cynamon', 'rodzynki'], 68, 'K', datetime.now() - timedelta(days=5)),
    (7, ['mąka_pszenna', 'proszek_do_pieczenia', 'cukier_puder', 'jajka', 'margaryna'], 68, 'K', datetime.now() - timedelta(days=1)),

    # User 8 - Starszy mężczyzna
    (8, ['kawa', 'chleb_razowy', 'wędlina_wieprzowa', 'ser_pleśniowy', 'oliwki', 'wino_czerwone'], 65, 'M', datetime.now() - timedelta(days=13)),
    (8, ['ser_pleśniowy', 'wino_biale', 'winogrona', 'szynka_surowa', 'bagietka'], 65, 'M', datetime.now() - timedelta(days=10)),
    (8, ['salami', 'ser_camembert', 'chleb_żytni', 'masło', 'rzodkiewka', 'piwo_jasne'], 65, 'M', datetime.now() - timedelta(days=7)),
    (8, ['wędzona_makrela', 'cebula', 'śmietana', 'chleb_baltonowski', 'koper'], 65, 'M', datetime.now() - timedelta(days=4)),
    (8, ['kawa_mielona', 'herbata_ceylon', 'ciastka_czekoladowe', 'mleko', 'cukier_brązowy'], 65, 'M', datetime.now() - timedelta(days=1)),

    # User 9 - Kobieta (rodzina)
    (9, ['mleko', 'chleb', 'masło', 'jajka', 'ser_żółty', 'wędlina', 'pomidory', 'ogórki', 'jabłka', 'banany'], 35, 'K', datetime.now() - timedelta(days=12)),
    (9, ['mleko', 'płatki_śniadaniowe', 'jogurt_naturalny', 'miód', 'migdały', 'orzechy'], 35, 'K', datetime.now() - timedelta(days=9)),
    (9, ['kurczak', 'ziemniaki', 'marchewka', 'cebula', 'pietruszka', 'koperek'], 35, 'K', datetime.now() - timedelta(days=7)),
    (9, ['makaron_spaghetti', 'sos_bolognese', 'ser_parmezan', 'sałata', 'pomidory', 'ogórki'], 35, 'K', datetime.now() - timedelta(days=4)),
    (9, ['jogurty_owocowe', 'chrupki_śniadaniowe', 'serek_wiejski', 'dżem_truskawkowy', 'bułki'], 35, 'K', datetime.now() - timedelta(days=1)),

    # User 10 - Mężczyzna (rodzina)
    (10, ['mąka', 'cukier', 'jajka', 'mleko', 'olej', 'drożdże', 'czekolada', 'orzechy'], 38, 'M', datetime.now() - timedelta(days=14)),
    (10, ['mąka_żytnia', 'siemie_lniane', 'płatki_owsiane', 'miód', 'jogurt_naturalny'], 38, 'M', datetime.now() - timedelta(days=11)),
    (10, ['wołowina', 'cebula', 'czosnek', 'pieczarki', 'śmietana_18%', 'makaron'], 38, 'M', datetime.now() - timedelta(days=8)),
    (10, ['łosoś', 'szpinak', 'ziemniaki', 'cytryna', 'koper', 'śmietana_12%'], 38, 'M', datetime.now() - timedelta(days=5)),
    (10, ['jajka', 'bekon', 'pomidory', 'pieczarki', 'ser_cheddar', 'tosty'], 38, 'M', datetime.now() - timedelta(days=2))
]

product_categories = {
    # 🥛 NABIAŁ I PRODUKTY MLECZNE
    "mleko": ["nabiał", "napoje"],
    "ser": ["nabiał", "białko"],
    "jogurt": ["nabiał", "przekąski"],
    "jogurt_naturalny": ["nabiał", "zdrowa_żywność"],
    "jogurt_owocowy": ["nabiał", "słodycze"],
    "jogurty_owocowe": ["nabiał", "słodycze"],
    "jogurt_grecki": ["nabiał", "zdrowa_żywność"],
    "jogurt_dla_dzieci": ["nabiał", "dzieci"],
    "śmietana": ["nabiał", "dodatki"],
    "śmietana_12%": ["nabiał", "dodatki"],
    "śmietana_18%": ["nabiał", "dodatki"],
    "masło": ["nabiał", "tłuszcze"],
    "masło_ekstra": ["nabiał", "tłuszcze"],
    "serek_wiejski": ["nabiał", "białko"],
    "serek_homogenizowany": ["nabiał", "słodycze"],
    
    # 🍞 PIECZYWO I WYPIEKI
    "chleb": ["pieczywo", "podstawowe"],
    "chleb_razowy": ["pieczywo", "zdrowa_żywność"],
    "chleb_żytni": ["pieczywo", "zdrowa_żywność"],
    "chleb_baltonowski": ["pieczywo", "podstawowe"],
    "bułki": ["pieczywo", "podstawowe"],
    "bułki_maślane": ["pieczywo", "słodycze"],
    "bagietka": ["pieczywo", "podstawowe"],
    "tosty": ["pieczywo", "śniadaniowe"],
    
    # 🥩 MIĘSO I WĘDLINY
    "kurczak": ["mięso", "białko"],
    "wołowina": ["mięso", "białko"],
    "wędlina": ["wędliny", "białko"],
    "wędlina_drobiowa": ["wędliny", "białko"],
    "wędlina_wieprzowa": ["wędliny", "białko"],
    "szynka": ["wędliny", "białko"],
    "szynka_surowa": ["wędliny", "białko"],
    "kielbasa": ["wędliny", "białko"],
    "parówki": ["wędliny", "fast_food"],
    "bekon": ["wędliny", "tłuszcze"],
    "boczek": ["wędliny", "tłuszcze"],
    "salami": ["wędliny", "przekąski"],
    
    # 🐟 RYBY I OWOCE MORZA
    "łosoś": ["ryby", "białko"],
    "tuńczyk": ["ryby", "konserwy"],
    "wędzona_makrela": ["ryby", "przekąski"],
    
    # 🥚 JAJKA
    "jajka": ["podstawowe", "białko"],
    
    # 🌾 PRODUKTY ZBOŻOWE
    "mąka": ["podstawowe", "pieczenie"],
    "mąka_pszenna": ["podstawowe", "pieczenie"],
    "mąka_żytnia": ["podstawowe", "zdrowa_żywność"],
    "ryż": ["podstawowe", "węglowodany"],
    "ryż_basmati": ["podstawowe", "węglowodany"],
    "ryż_jaśminowy": ["podstawowe", "węglowodany"],
    "ryż_okrągły": ["podstawowe", "węglowodany"],
    "makaron": ["podstawowe", "węglowodany"],
    "makaron_spaghetti": ["podstawowe", "węglowodany"],
    "makaron_penne": ["podstawowe", "węglowodany"],
    "płatki_śniadaniowe": ["śniadaniowe", "węglowodany"],
    "płatki_owsiane": ["śniadaniowe", "zdrowa_żywność"],
    "kasza": ["podstawowe", "zdrowa_żywność"],
    
    # 🥬 WARZYWA
    "pomidory": ["warzywa", "świeże"],
    "ogórki": ["warzywa", "świeże"],
    "marchew": ["warzywa", "świeże"],
    "marchewka": ["warzywa", "świeże"],
    "cebula": ["warzywa", "przyprawy"],
    "czosnek": ["warzywa", "przyprawy"],
    "papryka": ["warzywa", "świeże"],
    "sałata": ["warzywa", "świeże"],
    "szpinak": ["warzywa", "zdrowa_żywność"],
    "brokuły": ["warzywa", "zdrowa_żywność"],
    "kalafior": ["warzywa", "zdrowa_żywność"],
    "ziemniaki": ["warzywa", "podstawowe"],
    "ziemniaki_młode": ["warzywa", "świeże"],
    "pieczarki": ["warzywa", "świeże"],
    "kukurydza": ["warzywa", "konserwy"],
    "groszek": ["warzywa", "konserwy"],
    "awokado": ["warzywa", "zdrowa_żywność"],
    "rzodkiewka": ["warzywa", "świeże"],
    "szparagi": ["warzywa", "świeże"],
    
    # 🍎 OWOCE
    "jabłka": ["owoce", "świeże"],
    "banany": ["owoce", "świeże"],
    "gruszki": ["owoce", "świeże"],
    "winogrona": ["owoce", "świeże"],
    "jagody": ["owoce", "świeże"],
    "cytryna": ["owoce", "przyprawy"],
    "limonka": ["owoce", "przyprawy"],
    
    # 🌿 PRZYPRAWY I ZIOŁA
    "sól": ["przyprawy", "podstawowe"],
    "pieprz": ["przyprawy", "podstawowe"],
    "curry": ["przyprawy", "egzotyczne"],
    "imbir": ["przyprawy", "egzotyczne"],
    "cynamon": ["przyprawy", "słodkie"],
    "liść_laurowy": ["przyprawy", "zioła"],
    "koperek": ["przyprawy", "zioła"],
    "kolendra": ["przyprawy", "zioła"],
    "bazylia": ["przyprawy", "zioła"],
    "pietruszka": ["przyprawy", "zioła"],
    "przyprawy": ["przyprawy", "podstawowe"],
    
    # 🍯 SŁODYCZE I PRZEKĄSKI
    "cukier": ["słodycze", "podstawowe"],
    "cukier_brązowy": ["słodycze", "zdrowa_żywność"],
    "cukier_puder": ["słodycze", "pieczenie"],
    "cukier_wanilia": ["słodycze", "pieczenie"],
    "cukier_wanilinowy": ["słodycze", "pieczenie"],
    "miód": ["słodycze", "zdrowa_żywność"],
    "dżem": ["słodycze", "śniadaniowe"],
    "dżem_truskawkowy": ["słodycze", "śniadaniowe"],
    "czekolada": ["słodycze", "przekąski"],
    "ciastka_maślane": ["słodycze", "przekąski"],
    "ciastka_czekoladowe": ["słodycze", "przekąski"],
    "herbatniki": ["słodycze", "przekąski"],
    "biszkopty": ["słodycze", "przekąski"],
    "wafelki": ["słodycze", "przekąski"],
    "lody": ["słodycze", "mrożonki"],
    "lody_czekoladowe": ["słodycze", "mrożonki"],
    
    # 🥤 NAPOJE
    "woda": ["napoje", "podstawowe"],
    "sok_jabłkowy": ["napoje", "soki"],
    "sok_pomarańczowy": ["napoje", "soki"],
    "kawa": ["napoje", "używki"],
    "kawa_ziarnista": ["napoje", "używki"],
    "kawa_mielona": ["napoje", "używki"],
    "herbata": ["napoje", "używki"],
    "herbata_owocowa": ["napoje", "używki"],
    "herbata_ceylon": ["napoje", "używki"],
    "cola": ["napoje", "gazowane"],
    "energetyk": ["napoje", "używki"],
    
    # 🍷 ALKOHOL
    "piwo": ["alkohol", "napoje"],
    "piwo_jasne": ["alkohol", "napoje"],
    "piwo_craft": ["alkohol", "napoje"],
    "wino_czerwone": ["alkohol", "napoje"],
    "wino_biale": ["alkohol", "napoje"],
    
    # 🍳 TŁUSZCZE I OLEJE
    "oliwa": ["tłuszcze", "zdrowa_żywność"],
    "olej": ["tłuszcze", "podstawowe"],
    "olej_sezamowy": ["tłuszcze", "egzotyczne"],
    "margaryna": ["tłuszcze", "podstawowe"],
    
    # 🍕 PRODUKTY GOTOWE I MROŻONKI
    "pizza_mrożona": ["mrożonki", "fast_food"],
    "mrożona_pizza": ["mrożonki", "fast_food"],
    "burger_mrożony": ["mrożonki", "fast_food"],
    "frytki": ["mrożonki", "fast_food"],
    "warzywa_mrożone": ["mrożonki", "warzywa"],
    
    # 🥫 SOSY I DODATKI
    "keczup": ["sosy", "podstawowe"],
    "majonez": ["sosy", "podstawowe"],
    "musztarda": ["sosy", "podstawowe"],
    "sos_pomidorowy": ["sosy", "podstawowe"],
    "sos_bolognese": ["sosy", "gotowe"],
    "sos_sojowy": ["sosy", "egzotyczne"],
    
    # 🥜 ORZECHY I NASIONA
    "orzechy": ["zdrowe_przekąski", "białko"],
    "orzeszki": ["zdrowe_przekąski", "przekąski"],
    "migdały": ["zdrowe_przekąski", "białko"],
    "siemie_lniane": ["zdrowe_przekąski", "zdrowa_żywność"],
    
    # 🍿 PRZEKĄSKI SŁONE
    "chipsy": ["przekąski", "słone"],
    "paluszki": ["przekąski", "słone"],
    "krakersy": ["przekąski", "słone"],
    "słone_przekąski": ["przekąski", "słone"],
    "chrupki": ["przekąski", "słone"],
    "chrupki_kukurydziane": ["przekąski", "słone"],
    "chrupki_śniadaniowe": ["przekąski", "śniadaniowe"],
    
    # 👶 ARTYKUŁY DLA DZIECI
    "pampersy": ["dzieci", "higiena"],
    "mus_owocowy": ["dzieci", "jedzenie"],
    "chusteczki_nawilżane": ["dzieci", "higiena"],
    
    # 🧼 ARTYKUŁY HIGIENICZNE
    "mydło": ["higiena", "podstawowe"],
    "chusteczki": ["higiena", "podstawowe"],
    
    # 🧁 SKŁADNIKI DO PIECZENIA
    "drożdże": ["pieczenie", "podstawowe"],
    "proszek_do_pieczenia": ["pieczenie", "podstawowe"],
    "wanilia": ["pieczenie", "przyprawy"],
    "rodzynki": ["pieczenie", "słodycze"],
    
    # 🌱 PRODUKTY ALTERNATYWNE
    "mleko_kokosowe": ["napoje_roślinne", "egzotyczne"],
    "mleko_migdałowe": ["napoje_roślinne", "zdrowa_żywność"]
}

list_categories = []
for data in users_data:
    categories = []
    for product in data[1]:
        categories.extend(product_categories.get(product, []))
    list_categories.append(list(categories))

def convert_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # "2024-01-29T12:00:00.000000"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

data_to_save = {
    "users_data": [
        {
            "user_id": user_id,
            "products": products,
            "categories": categories,
     #       "age": age,
#            "gender": gender,
            "timestamp": timestamp
        }
        for (user_id, products, age, gender, timestamp),categories in zip(users_data,list_categories)
    ],
    "metadata": {
        "total_sessions": len(users_data),
        "unique_users": len(set(user_id for user_id, _, _, _, _ in users_data)),
        "generated_at": datetime.now().isoformat()
    }
}
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"
file_path = data_dir / "example_input.json"

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=2, default=convert_datetime)
print(data_to_save)