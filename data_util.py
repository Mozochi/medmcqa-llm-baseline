import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_file(dir):
    return json.loads(dir)

def json_to_dataframe(dir):
    data = [] # List to append all JSON objects to

    with open(dir, 'r') as f:
        for line in f:
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line.strip()}. Error: {e}")

    if data: 
        df = pd.DataFrame(data) # Returning DataFrame with parsed json data
    else:
        df = pd.DataFrame() # Returning an empty DataFrame if no valid data
    return df

def load_and_transform_json(dir, **kwargs): # dir: directory of json data,
    df = json_to_dataframe(dir) # Calling json_to_dataframe to load json data into a DataFrame

    for key, value in kwargs.items():
        try:
            match key:
                case "drop": # Used to drop columns taking in the column titles
                    for title in value:
                        df =df.drop(value, axis=1)

                case "match_rows": # Used to only pass through rows that contain a keyword
                    subject_df = df['subject_name'].str.contains(value, case=False, na=False) 
                    subject_df = df[subject_df]  
                    return subject_df
                    
                        
        
            


        except Exception as e:
            print(f"Error transforming database with key: {key}, value: {value}. Error: {e}")
    
    return df


def df_to_parquet(df_data, file_name):
    parquet_table = pa.Table.from_pandas(df_data)
    pq.write_table(parquet_table, f'{file_name}.parquet')


# Only used for debugging
def _df_to_csv(df_data, file_name): 
    csv_data = df_data.to_csv(f'{file_name}.csv', sep=',')
