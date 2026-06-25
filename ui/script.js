// Comprehensive Amenities List with Icons
const AMENITIES = [
    { label: "📶 WiFi", val: "has_wifi" },
    { label: "❄️ Klima (AC)", val: "has_ac" },
    { label: "📺 TV", val: "has_tv" },
    { label: "🍳 Mutfak", val: "has_kitchen" },
    { label: "🧺 Çamaşır Mak.", val: "has_washer" },
    { label: "🛗 Asansör", val: "has_elevator" },
    { label: "🅿️ Ücretsiz Otopark", val: "has_parking" },
    { label: "🏊 Özel Havuz", val: "has_pool" },
    { label: "🏋️ Spor Salonu", val: "has_gym" },
    { label: "🫧 Jakuzi", val: "has_jacuzzi" },
    { label: "🌅 Balkon / Teras", val: "has_balcony" }
];

let lastPredictedPrice = 0;

// Debounce Utility for Live Calculation
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const triggerPrediction = debounce(predictPrice, 280);

// Initialize Amenities Pills
const amenitiesGrid = document.getElementById('amenities-grid');

AMENITIES.forEach(amenity => {
    const pill = document.createElement('div');
    pill.className = 'amenity-pill';
    pill.innerText = amenity.label;
    pill.setAttribute('data-val', amenity.val);
    
    // Default active some basic amenities
    if(["has_wifi", "has_kitchen", "has_tv", "has_ac"].includes(amenity.val)) {
        pill.classList.add('active');
    }

    pill.addEventListener('click', () => {
        pill.classList.toggle('active');
        triggerPrediction();
    });

    amenitiesGrid.appendChild(pill);
});

// Slider Badge Live Updates & Event Listeners
const sliders = [
    { id: 'accommodates', unit: ' Kişi' },
    { id: 'bedrooms', unit: ' Oda' },
    { id: 'bathrooms', unit: ' Banyo' },
    { id: 'minimum_nights', unit: ' Gece' },
    { id: 'number_of_reviews', unit: ' Yorum' },
    { id: 'review_scores_rating', unit: ' ⭐' }
];

sliders.forEach(s => {
    const el = document.getElementById(s.id);
    const badge = document.getElementById(`badge-${s.id}`);
    
    el.addEventListener('input', (e) => {
        badge.innerText = e.target.value + s.unit;
        triggerPrediction();
    });
});

// Dropdowns and Checkboxes Event Listeners
['neighborhood', 'room_type'].forEach(id => {
    document.getElementById(id).addEventListener('change', triggerPrediction);
});

['is_superhost', 'instant_bookable'].forEach(id => {
    document.getElementById(id).addEventListener('change', triggerPrediction);
});

// Neighborhood Geo Info Updates
const NEIGH_GEO = {
    'Besiktas': "Merkeze: ~3.8 km | Boğaz'a: ~0.5 km (Yüksek Prim)",
    'Beyoglu': "Merkeze: ~0.8 km | Boğaz'a: ~1.5 km (Turistik Merkez)",
    'Sisli': "Merkeze: ~2.2 km | Boğaz'a: ~2.8 km (İş & Alışveriş)",
    'Kadikoy': "Merkeze: ~8.5 km | Boğaz'a: ~1.8 km (Anadolu Kültür Hattı)",
    'Fatih': "Merkeze: ~3.5 km | Boğaz'a: ~2.1 km (Tarihi Yarımada)",
    'Sariyer': "Merkeze: ~14.0 km | Boğaz'a: ~0.2 km (Boğaz Manzara Primi)",
    'Bakirkoy': "Merkeze: ~11.0 km | Sahile: ~0.8 km",
    'Atasehir': "Merkeze: ~12.5 km | Finans Merkezi",
    'Uskudar': "Merkeze: ~6.0 km | Boğaz'a: ~0.4 km",
    'Other': "İstanbul Geneli Ortalama Referans Değerleri"
};

document.getElementById('neighborhood').addEventListener('change', (e) => {
    const info = NEIGH_GEO[e.target.value] || "İstanbul Geneli Standart Değerleme";
    document.getElementById('geo-info').innerText = info;
});

// Smooth Rolling Number Animation
function animateValue(obj, start, end, duration) {
    if (start === end) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        // Ease out cubic
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const currentVal = Math.floor(easeProgress * (end - start) + start);
        obj.innerText = currentVal.toLocaleString('tr-TR');
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.innerText = end.toLocaleString('tr-TR');
        }
    };
    window.requestAnimationFrame(step);
}

// Core Prediction API Call
async function predictPrice() {
    // Collect boolean flags
    const getFlag = (val) => {
        const p = document.querySelector(`.amenity-pill[data-val="${val}"]`);
        return (p && p.classList.contains('active')) ? 1 : 0;
    };

    const payload = {
        neighborhood: document.getElementById('neighborhood').value,
        room_type: document.getElementById('room_type').value,
        accommodates: document.getElementById('accommodates').value,
        bedrooms: document.getElementById('bedrooms').value,
        bathrooms: document.getElementById('bathrooms').value,
        minimum_nights: document.getElementById('minimum_nights').value,
        number_of_reviews: document.getElementById('number_of_reviews').value,
        review_scores_rating: document.getElementById('review_scores_rating').value,
        is_superhost: document.getElementById('is_superhost').checked,
        instant_bookable: document.getElementById('instant_bookable').checked,
        has_wifi: getFlag('has_wifi'),
        has_ac: getFlag('has_ac'),
        has_tv: getFlag('has_tv'),
        has_kitchen: getFlag('has_kitchen'),
        has_washer: getFlag('has_washer'),
        has_elevator: getFlag('has_elevator'),
        has_parking: getFlag('has_parking'),
        has_pool: getFlag('has_pool'),
        has_gym: getFlag('has_gym'),
        has_jacuzzi: getFlag('has_jacuzzi'),
        has_balcony: getFlag('has_balcony')
    };

    try {
        const res = await fetch('http://127.0.0.1:5001/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("API Network Error");
        const data = await res.json();

        // 1. Animate Price Number
        const priceEl = document.getElementById('predicted-price');
        animateValue(priceEl, lastPredictedPrice, data.predicted_price, 500);
        lastPredictedPrice = data.predicted_price;

        // 2. Update Bounds
        document.getElementById('margin-error').innerText = data.margin_of_error.toLocaleString('tr-TR');
        document.getElementById('lower-bound').innerText = data.lower_bound.toLocaleString('tr-TR') + ' ₺';
        document.getElementById('upper-bound').innerText = data.upper_bound.toLocaleString('tr-TR') + ' ₺';

        // 3. Populate Visual SHAP Bars
        const shapContainer = document.getElementById('shap-list');
        shapContainer.innerHTML = '';

        // Find max magnitude for relative scaling
        const maxImpact = Math.max(...data.shap_factors.map(f => Math.abs(f.impact_magnitude)), 1);

        data.shap_factors.forEach(factor => {
            const row = document.createElement('div');
            row.className = 'shap-row';

            const isPos = factor.direction === 'positive';
            const percent = Math.min(Math.max((Math.abs(factor.impact_magnitude) / maxImpact) * 100, 15), 100);

            row.innerHTML = `
                <div class="shap-label-row">
                    <span class="shap-feature-name">${factor.feature}</span>
                    <span class="${isPos ? 'shap-val-pos' : 'shap-val-neg'}">
                        ${isPos ? '+ Artırdı ⬆' : '- Düşürdü ⬇'}
                    </span>
                </div>
                <div class="shap-bar-track">
                    <div class="${isPos ? 'shap-bar-pos' : 'shap-bar-neg'}" style="width: ${percent}%"></div>
                </div>
            `;

            shapContainer.appendChild(row);
        });

    } catch (err) {
        console.error("Prediction Error:", err);
    }
}

// Initial calculation on load
window.addEventListener('DOMContentLoaded', () => {
    predictPrice();
});
