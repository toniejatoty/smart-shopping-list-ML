from datetime import datetime, timedelta
import json
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

def convert_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # "2024-01-29T12:00:00.000000"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
data_to_save = {
    "users_data": [
        {
            "user_id": user_id,
            "products": products,
            "age": age,
            "gender": gender,
            "timestamp": timestamp
        }
        for user_id, products, age, gender, timestamp in users_data
    ],
    "metadata": {
        "total_sessions": len(users_data),
        "unique_users": len(set(user_id for user_id, _, _, _, _ in users_data)),
        "generated_at": datetime.now().isoformat()
    }
}
with open('example_input.json', 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=2, default=convert_datetime)