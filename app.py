import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='ui')
CORS(app)

# Load models
model_entire_home = joblib.load('model_entire_home_tuned.pkl')
model_rooms = joblib.load('model_rooms_tuned.pkl')

# Expected feature lists from the trained models
FEATURES_ENTIRE_HOME = list(model_entire_home.feature_names_in_)
FEATURES_ROOMS = list(model_rooms.feature_names_in_)

# Valid neighborhoods and their exact center coordinates
NEIGHBORHOOD_COORDS = {
    'Besiktas': (41.0422, 29.0083),
    'Kadikoy': (40.9901, 29.0292),
    'Sisli': (41.0613, 28.9877),
    'Beyoglu': (41.0369, 28.9775),
    'Fatih': (41.0156, 28.9536),
    'Bakirkoy': (40.9833, 28.8681),
    'Esenyurt': (41.0343, 28.6801),
    'Maltepe': (40.9248, 29.1311),
    'Atasehir': (40.9850, 29.1083),
    'Uskudar': (41.0269, 29.0158),
    'Kagithane': (41.0772, 28.9786),
    'Basaksehir': (41.0827, 28.7963),
    'Avcilar': (40.9801, 28.7175),
    'Beylikduzu': (41.0006, 28.6413),
    'Bagcilar': (41.0336, 28.8344),
    'Bahcelievler': (41.0031, 28.8465),
    'Kucukcekmece': (41.0003, 28.7844),
    'Umraniye': (41.0256, 29.0991),
    'Arnavutkoy': (41.1837, 28.7423),
    'Sile': (41.1754, 29.6133),
    'Other': (41.0082, 28.9784) # Default to generic center
}

NEIGHBORHOODS = list(NEIGHBORHOOD_COORDS.keys())

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('ui', path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    room_type = data.get('room_type', 'Entire home/apt')
    is_entire_home = 1 if room_type == 'Entire home/apt' else 0
    
    EXPECTED_FEATURES = FEATURES_ENTIRE_HOME if is_entire_home else FEATURES_ROOMS
    feature_dict = {f: 0.0 for f in EXPECTED_FEATURES}
    
    # 1. Location & Geography (from Neighborhood Center)
    neighborhood = data.get('neighborhood', 'Other')
    if neighborhood not in NEIGHBORHOODS: neighborhood = 'Other'
    
    lat, lon = NEIGHBORHOOD_COORDS[neighborhood]
    
    if 'latitude' in feature_dict: feature_dict['latitude'] = lat
    if 'longitude' in feature_dict: feature_dict['longitude'] = lon
    
    taksim_lat, taksim_lon = 41.0370, 28.9850
    bosphorus_lat, bosphorus_lon = 41.0436, 29.0306
    
    if 'distance_to_center_km' in feature_dict: feature_dict['distance_to_center_km'] = haversine(lat, lon, taksim_lat, taksim_lon)
    if 'distance_to_bosphorus_km' in feature_dict: feature_dict['distance_to_bosphorus_km'] = haversine(lat, lon, bosphorus_lat, bosphorus_lon)
    
    # Neighborhood One-Hot
    if f'neigh_{neighborhood}' in feature_dict: feature_dict[f'neigh_{neighborhood}'] = 1.0
    
    # Density heuristic
    central = ['Besiktas', 'Beyoglu', 'Fatih', 'Kadikoy', 'Sisli']
    if 'listing_density' in feature_dict: feature_dict['listing_density'] = 1500.0 if neighborhood in central else 300.0
    
    # 2. Property Basics
    accommodates = int(data.get('accommodates', 2))
    bedrooms = int(data.get('bedrooms', 1))
    bathrooms = float(data.get('bathrooms', 1.0))
    
    if 'accommodates' in feature_dict: feature_dict['accommodates'] = accommodates
    if 'bedrooms' in feature_dict: feature_dict['bedrooms'] = bedrooms
    if 'beds' in feature_dict: feature_dict['beds'] = accommodates
    if 'bathrooms_num' in feature_dict: feature_dict['bathrooms_num'] = bathrooms
    
    # Amenities Booleans
    boolean_amenities = [
        'has_wifi', 'has_ac', 'has_tv', 'has_kitchen', 'has_washer', 
        'has_elevator', 'has_parking', 'has_pool', 'has_gym', 'has_jacuzzi', 'has_balcony'
    ]
    for am in boolean_amenities:
        if am in feature_dict:
            feature_dict[am] = int(data.get(am, 0))
    
    # Room Type One-Hot
    if 'room_type_Entire home/apt' in feature_dict: feature_dict['room_type_Entire home/apt'] = is_entire_home
    if 'room_type_Private room' in feature_dict: feature_dict['room_type_Private room'] = 1 if not is_entire_home else 0
    
    # Property Type default
    if is_entire_home:
        if 'prop_type_Entire rental unit' in feature_dict: feature_dict['prop_type_Entire rental unit'] = 1.0
    else:
        if 'prop_type_Private room in rental unit' in feature_dict: feature_dict['prop_type_Private room in rental unit'] = 1.0

    # 3. Host & Booking Settings
    if 'host_is_superhost' in feature_dict: feature_dict['host_is_superhost'] = 1.0 if data.get('is_superhost') else 0.0
    if 'instant_bookable' in feature_dict: feature_dict['instant_bookable'] = 1.0 if data.get('instant_bookable') else 0.0
    if 'minimum_nights' in feature_dict: feature_dict['minimum_nights'] = int(data.get('minimum_nights', 1))
    if 'minimum_minimum_nights' in feature_dict: feature_dict['minimum_minimum_nights'] = feature_dict['minimum_nights']
    if 'maximum_nights' in feature_dict: feature_dict['maximum_nights'] = 365
    
    # 4. Reviews & Quality
    reviews = int(data.get('number_of_reviews', 0))
    rating = float(data.get('review_scores_rating', 0.0))
    
    if 'number_of_reviews' in feature_dict: feature_dict['number_of_reviews'] = reviews
    if 'has_reviews' in feature_dict: feature_dict['has_reviews'] = 1.0 if reviews > 0 else 0.0
    if 'review_scores_rating' in feature_dict: feature_dict['review_scores_rating'] = rating
    if 'reviews_per_month' in feature_dict: feature_dict['reviews_per_month'] = min(reviews / 12.0, 5.0) if reviews > 0 else 0.0
    
    if reviews > 0:
        if 'review_scores_accuracy' in feature_dict: feature_dict['review_scores_accuracy'] = rating
        if 'review_scores_cleanliness' in feature_dict: feature_dict['review_scores_cleanliness'] = rating
        if 'review_scores_checkin' in feature_dict: feature_dict['review_scores_checkin'] = rating
        if 'review_scores_communication' in feature_dict: feature_dict['review_scores_communication'] = rating
        if 'review_scores_location' in feature_dict: feature_dict['review_scores_location'] = rating
        if 'review_scores_value' in feature_dict: feature_dict['review_scores_value'] = rating
    
    # Create dataframe
    df = pd.DataFrame([feature_dict])[EXPECTED_FEATURES]
    
    # Choose Model
    model = model_entire_home if is_entire_home else model_rooms
    mae = 1177 if is_entire_home else 654
    
    # Predict
    pred_log = model.predict(df)[0]
    pred_price = np.expm1(pred_log)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    
    # Extract top 3 features
    importances = shap_values[0]
    top_indices = np.argsort(np.abs(importances))[-15:][::-1]
    
    top_factors = []
    friendly_names = {
        'distance_to_bosphorus_km': "🌊 Boğaz'a Uzaklık",
        'distance_to_center_km': "📍 Merkeze Uzaklık",
        'accommodates': "👥 Misafir Kapasitesi",
        'bedrooms': "🛏️ Yatak Odası Sayısı",
        'beds': "🛌 Yatak Sayısı",
        'bathrooms_num': "🚿 Banyo Sayısı",
        'listing_density': "🏙️ Semt Turistik Yoğunluğu",
        'has_wifi': "📶 WiFi Olanakları",
        'has_ac': "❄️ Klima (AC)",
        'has_tv': "📺 Televizyon",
        'has_kitchen': "🍳 Mutfak",
        'has_washer': '🧺 Çamaşır Makinesi',
        'has_elevator': "🛗 Asansör",
        'has_parking': "🅿️ Ücretsiz Otopark",
        'has_pool': "🏊 Özel Havuz",
        'has_gym': "🏋️ Spor Salonu",
        'has_jacuzzi': "🫧 Jakuzi",
        'has_balcony': "🌅 Balkon / Teras",
        'host_is_superhost': "🎖️ Süperhost Statüsü",
        'instant_bookable': "⚡ Anında Rezervasyon",
        'number_of_reviews': "💬 Toplam Yorum Sayısı",
        'has_reviews': "📝 Yorum Varlığı",
        'reviews_per_month': "📈 Aylık Yorum Sıklığı",
        'review_scores_rating': "⭐ Ortalama Puan",
        'review_scores_accuracy': "🎯 Doğruluk Puanı",
        'review_scores_cleanliness': "✨ Temizlik Puanı",
        'review_scores_checkin': "🔑 Giriş Kolaylığı Puanı",
        'review_scores_communication': "📞 İletişim Puanı",
        'review_scores_location': "📍 Lokasyon Beğeni Puanı",
        'review_scores_value': "💰 Fiyat/Performans Puanı",
        'minimum_nights': "🌙 Minimum Konaklama",
        'minimum_minimum_nights': "🌙 Alt Konaklama Sınırı",
        'maximum_nights': "📅 Maksimum Konaklama",
        'latitude': "🧭 Enlem Koordinatı",
        'longitude': "🧭 Boylam Koordinatı"
    }
    
    for idx in top_indices:
        feat_name = EXPECTED_FEATURES[idx]
        val = importances[idx]
        if feat_name.startswith('tfidf_') or feat_name.startswith('neigh_'): 
            if feat_name.startswith('neigh_') and val != 0:
                friendly_names[feat_name] = f"📍 Semt Lokasyon Primi ({feat_name.split('_')[1]})"
            else:
                continue
            
        if val == 0: continue
        
        direction = "positive" if val > 0 else "negative"
        if feat_name.startswith('room_type_'):
            display_name = "🏡 Konaklama Tipi Tercihi"
        elif feat_name.startswith('prop_type_'):
            display_name = "🏗️ Ev Mimarisi Türü"
        elif feat_name.startswith('availability') or feat_name == 'has_availability':
            display_name = "📅 Takvim Müsaitlik Dinamiği"
        elif feat_name.startswith('number_of_reviews'):
            display_name = "💬 İlan Yorum Yoğunluğu"
        elif 'host_listings' in feat_name:
            display_name = "🔑 Ev Sahibi Portföy Genişliği"
        elif feat_name.startswith('host_response') or feat_name.startswith('host_acceptance'):
            display_name = "📞 Ev Sahibi İletişim Performansı"
        elif feat_name.startswith('desc_'):
            display_name = "📝 İlan Tanıtım Detay Kalitesi"
        elif feat_name.startswith('days_since'):
            display_name = "⏳ Platform & İlan Kıdemi"
        elif feat_name in friendly_names:
            display_name = friendly_names[feat_name]
        else:
            clean = feat_name.replace('_', ' ').title()
            display_name = f"📊 {clean}"
        top_factors.append({
            "feature": display_name,
            "direction": direction,
            "impact_magnitude": float(val)
        })
        if len(top_factors) == 4: break

    return jsonify({
        'predicted_price': round(pred_price),
        'margin_of_error': mae,
        'lower_bound': max(0, round(pred_price - mae)),
        'upper_bound': round(pred_price + mae),
        'shap_factors': top_factors
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
