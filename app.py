from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import re, random, pandas as pd, numpy as np, csv, warnings, requests
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from difflib import get_close_matches

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)
app.secret_key = "supersecret"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ------------------ Load Data ------------------
training = pd.read_csv('Data/Training.csv')
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# ------------------ Model Evaluation ------------------
y_pred = model.predict(x_test)
print("\n📊 MODEL PERFORMANCE")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred))

# ------------------ Dictionaries ------------------
severityDictionary, description_list, precautionDictionary = {}, {}, {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

with open('MasterData/symptom_Description.csv') as f:
    for row in csv.reader(f):
        description_list[row[0]] = row[1]

with open('MasterData/symptom_severity.csv') as f:
    for row in csv.reader(f):
        try:
            severityDictionary[row[0]] = int(row[1])
        except:
            pass

with open('MasterData/symptom_precaution.csv') as f:
    for row in csv.reader(f):
        precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# ------------------ Symptom NLP ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain"
}

def extract_symptoms(text, all_symptoms):
    extracted = []
    text = text.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_"," ") in text:
            extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_"," ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_"," ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

# ------------------ Prediction ------------------
def predict_disease(symptoms):
    vec = np.zeros(len(symptoms_dict))
    for s in symptoms:
        if s in symptoms_dict:
            vec[symptoms_dict[s]] = 1
    proba = model.predict_proba([vec])[0]
    idx = np.argmax(proba)
    disease = le.inverse_transform([idx])[0]
    confidence = round(proba[idx]*100,2)
    return disease, confidence, proba

def top_3_diseases(proba):
    top = np.argsort(proba)[-3:][::-1]
    return [(le.inverse_transform([i])[0], round(proba[i]*100,2)) for i in top]

# ------------------ Risk Score & Doctor ------------------
specialist_map = {
    "Diabetes": "Endocrinologist",
    "GERD": "Gastroenterologist",
    "Migraine": "Neurologist",
    "Asthma": "Pulmonologist",
    "Heart attack": "Cardiologist",
    "Depression": "Psychiatrist"
}

def calculate_risk(severity, days, confidence):
    risk = (severity*5) + (days*3) + (confidence*0.4)
    return min(int(risk),100)

# ------------------ Emergency Keywords ------------------
emergency_keywords = [
    "chest pain", "loss of consciousness",
    "severe bleeding", "seizure",
    "paralysis", "difficulty speaking"
]

# ------------------ Nearby Hospitals ------------------
def get_nearby_hospitals(lat, lon, api_key):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"location":f"{lat},{lon}","radius":3000,"type":"hospital","key":api_key}
    res = requests.get(url, params=params).json()
    return [h["name"] for h in res.get("results", [])[:3]]

# ------------------ Quotes ------------------
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ Routes ------------------
@app.route('/')
def index():
    session.clear()
    session['step'] = 'welcome'
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json['message']
    lat = request.json.get('lat')
    lon = request.json.get('lon')
    step = session.get('step', 'welcome')

    # Emergency Detection
    if any(word in msg.lower() for word in emergency_keywords):
        session['step'] = 'end'
        return jsonify(reply="🚑 EMERGENCY DETECTED! Go to nearest hospital immediately.")

    # Chat Steps
    if step == 'welcome':
        session['step'] = 'name'
        return jsonify(reply="🤖 Welcome! What's your name?")

    if step == 'name':
        session['name'] = msg
        session['step'] = 'age'
        return jsonify(reply="👉 Enter your age:")

    if step == 'age':
        try: session['age'] = int(msg)
        except: return jsonify(reply="❌ Enter a valid number.")
        session['step'] = 'gender'
        return jsonify(reply="👉 Gender? (M/F/Other)")

    if step == 'gender':
        session['gender'] = msg
        session['step'] = 'symptoms'
        return jsonify(reply="👉 Describe your symptoms:")

    if step == 'symptoms':
        syms = extract_symptoms(msg, cols)
        if not syms: return jsonify(reply="❌ Could not detect symptoms. Try again.")
        session['symptoms'] = syms
        session['step'] = 'days'
        return jsonify(reply=f"✅ Detected symptoms: {', '.join(syms)}\n👉 For how many days?")

    if step == 'days':
        try: session['days'] = int(msg)
        except: return jsonify(reply="❌ Enter a valid number.")
        session['step'] = 'severity'
        return jsonify(reply="👉 Rate severity 1–10:")

    if step == 'severity':
        try: session['severity'] = int(msg)
        except: return jsonify(reply="❌ Enter a valid number.")
        return final_prediction(lat, lon)

    if step == 'progress':
        msg_lower = msg.lower()
        if "worse" in msg_lower:
            reply = "🚨 Symptoms worsening! Consult a doctor immediately."
        elif "better" in msg_lower:
            reply = "✅ Symptoms improving. Continue precautions and rest."
        else:
            reply = "⚠️ Monitor symptoms closely for the next 24 hours."
        session['step'] = 'end'
        return jsonify(reply=reply)

# ------------------ Final Prediction ------------------
def final_prediction(lat=None, lon=None):
    disease, conf, proba = predict_disease(session['symptoms'])
    top3 = top_3_diseases(proba)
    text = f"🩺 Disease: {disease}\n🔎 Confidence: {conf}%\n\nTop 3 Conditions:\n"
    for d, c in top3: text += f"- {d} ({c}%)\n"

    # Risk Score
    risk = calculate_risk(session['severity'], session['days'], conf)
    text += f"\n🧮 Risk Score: {risk}/100"
    if risk>=70: text+="\n🔴 HIGH"
    elif risk>=40: text+="\n🟠 MEDIUM"
    else: text+="\n🟢 LOW"

    # Doctor Recommendation
    specialist = specialist_map.get(disease,"General Physician")
    text += f"\n👨‍⚕️ Recommended Specialist: {specialist}"

    # Severity Warning
    if session['severity'] >= 8 or session['days'] >= 7:
        text += "\n🚨 Symptoms severe or prolonged. Consult a doctor!"

    # Description + Precautions
    text += f"\n📖 About: {description_list.get(disease,'No description available.')}\n"
    if disease in precautionDictionary:
        text += "\n🛡️ Precautions:\n"
        for i,p in enumerate(precautionDictionary[disease],1): text += f"{i}. {p}\n"

    # Nearby Hospitals
    if lat and lon:
        hospitals = get_nearby_hospitals(lat, lon, api_key="YOUR_GOOGLE_API_KEY")
        if hospitals:
            text += "\n🏥 Nearby Hospitals:\n" + "\n".join(f"- {h}" for h in hospitals)

    # Motivational Quote
    text += "\n💡 " + random.choice(quotes)
    text += f"\n\nTake care, {session['name']} ❤️"

    # Ask about symptom progression
    session['step'] = 'progress'
    return jsonify(reply=text + "\n\n📊 Are your symptoms improving, worsening, or unchanged?")

if __name__ == "__main__":
    app.run(debug=True)
