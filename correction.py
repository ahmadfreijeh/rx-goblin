import pandas as pd
from rapidfuzz import process

def load_drug_list():
    path = "data/raw/fda/product.txt"
    df = pd.read_csv(path, sep="\t", usecols=["NONPROPRIETARYNAME"], encoding="latin-1")
    drug_list = df["NONPROPRIETARYNAME"].dropna().unique().tolist() # convert to list of unique drug names, drop any missing values
    return drug_list

def correct_drug_name(text, drug_list):
    match, score, _ = process.extractOne(text, drug_list)
    return match if score > 80 else text
