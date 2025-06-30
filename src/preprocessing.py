import pandas as pd
import numpy as np

def preprocess_bangalore_data(df):
    # Drop rows with critical nulls
    df = df.dropna(subset=['location', 'size', 'total_sqft', 'bath', 'price'])

    # Extract BHK from size column
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)

    # Clean total_sqft: handle ranges and non-numeric
    def convert_sqft(val):
        try:
            if '-' in val:
                tokens = val.split('-')
                return (float(tokens[0]) + float(tokens[1])) / 2
            return float(val)
        except:
            return None

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df = df.dropna(subset=['total_sqft'])

    # Outlier filtering
    df = df[df['total_sqft'] / df['bhk'] >= 300]
    df = df[df['bath'] <= df['bhk'] + 2]

    # Encode availability: 1 if ready-to-move or immediate possession
    df['availability_flag'] = df['availability'].apply(
        lambda x: 1 if isinstance(x, str) and ('ready' in x.lower() or 'immediate' in x.lower()) else 0
    )

    # Drop unused columns
    df = df.drop(columns=['size', 'availability', 'society'], errors='ignore')

    return df
