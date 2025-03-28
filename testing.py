from data_util import json_to_dataframe, load_and_transform_json, df_to_parquet, _df_to_csv
import pandas as pd

#print(json_to_dataframe(r".\MedMCQA\test.json"))

df = load_and_transform_json(r".\MedMCQA\test.json", match_rows=["Pediatrics", "Microbiology"])

_df_to_csv(df, "testing_csv")

#pd.read_parquet("file_name.parquet").head()
