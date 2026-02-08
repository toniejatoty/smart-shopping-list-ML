import pandas as pd
import ast
from pathlib import Path
import prepare_data_mapping_name_id

project_root = Path(__file__).parent.parent
model_input_path = project_root / "ml" / "data" / "model_input.csv"

if model_input_path.exists():
    print("Wczytywanie df z pliku")
    df = pd.read_csv(model_input_path)

    df['off_product_id'] = df['off_product_id'].apply(ast.literal_eval)
    df['mapped_categories'] = df['mapped_categories'].apply(ast.literal_eval)
else:
    print("Plik nie istnieje. Rozpoczynam mapowanie...")
    kaggle_processed = pd.read_csv(project_root / "ml" / "data" / "kaggle_prepared.csv")
    off_data = pd.read_csv(project_root / "ml" / "data" / "OpenFoodData.csv")
    
    p_mapped, c_mapped, product_map, category_map = prepare_data_mapping_name_id.map_products_and_categories(kaggle_processed, off_data)
    df = prepare_data_mapping_name_id.make_model_input_dataframe(kaggle_processed, p_mapped, c_mapped)
    
    df.to_csv(model_input_path, index=False)
