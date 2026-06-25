import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")

    # ---------------------------------------------------------
    # 1. Drop Scraping Metadata & IDs
    # ---------------------------------------------------------
    cols_to_drop = [
        'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 
        'picture_url', 'host_id', 'host_url', 'host_thumbnail_url', 
        'host_picture_url', 'calendar_last_scraped', 'calendar_updated',
        'license', 'host_name', 'host_neighbourhood', 'neighbourhood', 
        'neighbourhood_group_cleansed', 'host_location', 'host_verifications'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ---------------------------------------------------------
    # 2. Prevent Data Leakage
    # ---------------------------------------------------------
    leakage_cols = [
        'estimated_revenue_l365d', 'estimated_occupancy_l365d',
        'availability_30', 'availability_60', 'availability_90'
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # ---------------------------------------------------------
    # 3. Clean Target Variable (price)
    # ---------------------------------------------------------
    if 'price' in df.columns:
        # Remove $ and commas, convert to float
        df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        # Drop rows where price is NaN or 0
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]

    # ---------------------------------------------------------
    # 4. Host Information Processing
    # ---------------------------------------------------------
    reference_date = pd.to_datetime('2026-01-01') # Arbitrary recent date
    
    if 'host_since' in df.columns:
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        df['days_since_host_joined'] = (reference_date - df['host_since']).dt.days
        df = df.drop(columns=['host_since'])

    # Rates: convert '98%' to 0.98
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
            # Impute missing with median
            df[col] = df[col].fillna(df[col].median())

    # Booleans
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'has_availability']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0)

    # ---------------------------------------------------------
    # 5. Property & Room Details
    # ---------------------------------------------------------
    # Bathrooms text (e.g. "1.5 shared baths", "1 bath")
    if 'bathrooms_text' in df.columns:
        df['bathrooms_num'] = df['bathrooms_text'].astype(str).str.extract(r'([\d\.]+)').astype(float)
        df['is_shared_bath'] = df['bathrooms_text'].astype(str).str.contains('shared', case=False, na=False).astype(int)
        df = df.drop(columns=['bathrooms_text', 'bathrooms'], errors='ignore') # raw bathrooms is usually null

    # Numerical imputations
    for col in ['accommodates', 'bedrooms', 'beds', 'bathrooms_num']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Amenities: list to count
    if 'amenities' in df.columns:
        def check_amenity(am_str, keywords):
            if pd.isnull(am_str) or am_str == 'nan': return 0
            am_str = str(am_str).lower()
            for k in keywords:
                if k in am_str: return 1
            return 0
        
        df['has_wifi'] = df['amenities'].apply(lambda x: check_amenity(x, ['wifi', 'internet']))
        df['has_ac'] = df['amenities'].apply(lambda x: check_amenity(x, ['air conditioning', 'klima', 'ac -']))
        df['has_tv'] = df['amenities'].apply(lambda x: check_amenity(x, ['tv', 'television']))
        df['has_kitchen'] = df['amenities'].apply(lambda x: check_amenity(x, ['kitchen', 'mutfak', 'cooking basics']))
        df['has_washer'] = df['amenities'].apply(lambda x: check_amenity(x, ['washer', 'çamaşır makinesi']))
        df['has_elevator'] = df['amenities'].apply(lambda x: check_amenity(x, ['elevator', 'asansör']))
        df['has_parking'] = df['amenities'].apply(lambda x: check_amenity(x, ['parking', 'otopark', 'garage']))
        df['has_pool'] = df['amenities'].apply(lambda x: check_amenity(x, ['pool', 'havuz']))
        df['has_gym'] = df['amenities'].apply(lambda x: check_amenity(x, ['gym', 'fitness', 'spor salonu']))
        df['has_jacuzzi'] = df['amenities'].apply(lambda x: check_amenity(x, ['hot tub', 'jacuzzi', 'jakuzi']))
        df['has_balcony'] = df['amenities'].apply(lambda x: check_amenity(x, ['balcony', 'patio', 'teras', 'terrace']))
        
        df = df.drop(columns=['amenities'])

    # Drop raw text columns that NLP already processed, BUT KEEP 'description' for TF-IDF Pipeline
    text_cols = ['name', 'neighborhood_overview', 'host_about']
    df = df.drop(columns=[c for c in text_cols if c in df.columns])

    # ---------------------------------------------------------
    # 6. Reviews & Ratings
    # ---------------------------------------------------------
    review_cols = [c for c in df.columns if c.startswith('review_scores_')]
    for col in review_cols:
        if col == 'review_scores_rating':
            df['has_reviews'] = df[col].notna().astype(int)
        df[col] = df[col].fillna(-1)

    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

    for col in ['first_review', 'last_review']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'days_since_{col}'] = (reference_date - df[col]).dt.days
            df[f'days_since_{col}'] = df[f'days_since_{col}'].fillna(-1) # -1 implies no review
            df = df.drop(columns=[col])

    # ---------------------------------------------------------
    # 7. Categorical Encoding (One-Hot)
    # ---------------------------------------------------------
    cat_cols = ['room_type', 'host_response_time']
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            dummies = dummies.astype(int)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

    # Keep top 15 property types and top 20 neighbourhoods, group rest to 'Other'
    if 'property_type' in df.columns:
        top_props = df['property_type'].value_counts().nlargest(15).index
        df['property_type'] = df['property_type'].where(df['property_type'].isin(top_props), 'Other')
        dummies = pd.get_dummies(df['property_type'], prefix='prop_type').astype(int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=['property_type'])

    if 'neighbourhood_cleansed' in df.columns:
        top_neigh = df['neighbourhood_cleansed'].value_counts().nlargest(20).index
        df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].where(df['neighbourhood_cleansed'].isin(top_neigh), 'Other')
        dummies = pd.get_dummies(df['neighbourhood_cleansed'], prefix='neigh').astype(int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=['neighbourhood_cleansed'])

    # (NLP Artifact cleaning removed, no TF-IDF columns to clean here)

    # Safety net: fill any lingering numeric NaNs with 0
    df = df.fillna(0)

    print(f"Final shape: {df.shape}")
    print(f"Saving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    clean_data('listings_with_nlp.csv', 'listings_cleaned.csv')
