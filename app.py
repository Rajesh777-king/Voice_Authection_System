from flask import Flask, render_template, request, jsonify
import os
import subprocess
import gc
from datetime import datetime

import librosa
import torch
import numpy as np

from pymongo import MongoClient
from werkzeug.utils import secure_filename

# ======================================================
# IMPORT MODELS SAFELY
# ======================================================

# ECAPA
try:

    from models.ecapa_model import (
        model as ecapa_model
    )

    print("ECAPA loaded successfully!")

except Exception as e:

    ecapa_model = None

    print("ECAPA load failed:", e)

# CTA-E-BRANCHFORMER
try:

    from models.cta_ebranchformer import (
        cta_model
    )

    print("CTA-E-Branchformer loaded successfully!")

except Exception as e:

    cta_model = None

    print("CTA model load failed:", e)

# ECAPA-CONFORMER
try:

    from models.conformer_voip_finetuned import (
        ecapa_conformer_model
    )

    print("ECAPA-Conformer loaded successfully!")

except Exception as e:

    ecapa_conformer_model = None

    print("ECAPA-Conformer load failed:", e)

# EBRANCHFORMER
try:

    from models.ebranchformer_voip_finetuned import (
        model as ebranchformer_model
    )

    print("EBranchformer loaded successfully!")

except Exception as e:

    ebranchformer_model = None

    print("EBranchformer load failed:", e)

# ======================================================
# ECAPA ZEROSHOT KATHBATH
# ======================================================

try:

    from models.ecapa_zeroshot_kathpath import (
        model as ecapa_kathbath_model
    )

    print("ECAPA Zeroshot Kathbath loaded successfully!")

except Exception as e:

    ecapa_kathbath_model = None

    print("ECAPA Kathbath load failed:", e)

# ======================================================
# FLASK APP
# ======================================================

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

RECORDINGS_FOLDER = 'static/recordings'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RECORDINGS_FOLDER,
    exist_ok=True
)

# ======================================================
# DEVICE
# ======================================================

device = torch.device("cpu")

# ======================================================
# MONGODB CONNECTION
# ======================================================

try:

    client = MongoClient(
        "mongodb://localhost:27017/",
        serverSelectionTimeoutMS=3000
    )

    client.server_info()

    db = client["speaker_recognition"]

    users_collection = db["users"]

    print("MongoDB Connected Successfully!")

except Exception as e:

    print("MongoDB Connection Failed!")

    print(e)

    users_collection = None

# ======================================================
# FFMPEG PATH
# ======================================================

FFMPEG_PATH = (
    r"C:\ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
)

if not os.path.exists(FFMPEG_PATH):

    print("FFMPEG not found!")

def extract_features(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=16000
    )

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=80,
        n_fft=512,
        hop_length=160,
        win_length=400
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel = (
        mel - np.mean(mel)
    ) / (
        np.std(mel) + 1e-9
    )

    # =====================================================
    # IMPORTANT
    # ECAPA MODEL EXPECTS:
    # (B, T, 80)
    # =====================================================

    mel = mel.T

    features = torch.FloatTensor(
        mel
    ).unsqueeze(0)

    print("FEATURE SHAPE:", features.shape)

    return features

# ======================================================
# AUDIO CONVERSION
# ======================================================

def convert_webm_to_wav(webm_path):

    wav_path = webm_path.replace(
        '.webm',
        '.wav'
    )

    command = [

        FFMPEG_PATH,

        "-y",

        "-i",

        webm_path,

        wav_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode != 0:

        return None

    if not os.path.exists(wav_path):

        return None

    return wav_path

# ======================================================
# GENERATE EMBEDDING
# ======================================================

def generate_embedding(
    audio_path,
    selected_model
):

    # ==========================================
    # MODEL VALIDATION
    # ==========================================

    if (
        selected_model == 'ecapa'
        and
        ecapa_model is None
    ):

        return None

    if (
        selected_model == 'cta'
        and
        cta_model is None
    ):

        return None

    if (
        selected_model == 'ecapa_conformer'
        and
        ecapa_conformer_model is None
    ):

        return None

    if (
        selected_model == 'ebranchformer'
        and
        ebranchformer_model is None
    ):

        return None
    if (
        selected_model == 'ecapa_kathbath'
        and
        ecapa_kathbath_model is None
    ):

        return None
    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================

    features = extract_features(audio_path)

    # ==========================================
    # ECAPA
    # ==========================================

    if selected_model == 'ecapa':

        with torch.no_grad():

            embedding = ecapa_model(features)

    # ==========================================
    # CTA MODEL
    # ==========================================

    elif selected_model == 'cta':

        with torch.no_grad():

            embedding = cta_model(features)

    # ==========================================
    # ECAPA CONFORMER
    # ==========================================

    elif selected_model == 'ecapa_conformer':

        with torch.no_grad():

            embedding = ecapa_conformer_model(features)

    # ==========================================
    # EBRANCHFORMER
    # ==========================================

    elif selected_model == 'ebranchformer':

        with torch.no_grad():

            embedding = ebranchformer_model(features)
    # ==========================================
    # ECAPA KATHBATH
    # ==========================================

    elif selected_model == 'ecapa_kathbath':

        with torch.no_grad():

            embedding = ecapa_kathbath_model(features)
    else:

        return None

    embedding = embedding.squeeze().cpu().tolist()

    torch.cuda.empty_cache()

    gc.collect()

    return embedding

# ======================================================
# COSINE SIMILARITY
# ======================================================

def compute_cosine_similarity(
    emb1,
    emb2
):

    emb1 = np.array(
        emb1,
        dtype=np.float32
    )

    emb2 = np.array(
        emb2,
        dtype=np.float32
    )

    # =====================================================
    # L2 NORMALIZATION
    # =====================================================

    emb1 = emb1 / (
        np.linalg.norm(emb1) + 1e-10
    )

    emb2 = emb2 / (
        np.linalg.norm(emb2) + 1e-10
    )

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    similarity = np.dot(
        emb1,
        emb2
    )

    return float(similarity)

# ======================================================
# WEBSITE ROUTES
# ======================================================

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/about')
def about():

    return render_template('about.html')

@app.route('/resources')
def resources():

    return render_template('resources.html')

@app.route('/register')
def register_page():

    return render_template('register.html')

@app.route('/verify')
def verify_page():

    return render_template('verify.html')

# ======================================================
# REGISTER USER
# ======================================================

@app.route(
    '/register',
    methods=['POST']
)
def register_user():

    if users_collection is None:

        return jsonify({

            'status': 'error',

            'message': 'MongoDB server not running'
        })

    if 'audio' not in request.files:

        return jsonify({

            'status': 'error',

            'message': 'No audio received'
        })

    audio = request.files['audio']

    if audio.filename == '':

        return jsonify({

            'status': 'error',

            'message': 'Invalid audio file'
        })

    name = request.form.get('name')

    user_id = request.form.get('user_id')

    selected_model = request.form.get('model')

    if not name or not user_id:

        return jsonify({

            'status': 'error',

            'message': 'Name and User ID required'
        })

    existing_user = users_collection.find_one({

        'user_id': user_id,

        'model': selected_model
    })

    if existing_user:

        return jsonify({

            'status': 'error',

            'message': 'User already registered with this model'
        })

    timestamp = datetime.now().strftime(
        '%Y%m%d_%H%M%S'
    )

    safe_user_id = secure_filename(
        user_id
    )

    filename = (
        f'{safe_user_id}_{timestamp}.webm'
    )

    webm_path = os.path.join(
        RECORDINGS_FOLDER,
        filename
    )

    audio.save(webm_path)

    wav_path = convert_webm_to_wav(
        webm_path
    )

    if wav_path is None:

        return jsonify({

            'status': 'error',

            'message': 'Audio conversion failed'
        })

    embedding = generate_embedding(

        wav_path,

        selected_model
    )

    if embedding is None:

        return jsonify({

            'status': 'error',

            'message': 'Model inference failed'
        })

    user_data = {

        "name": name,

        "user_id": user_id,

        "model": selected_model,

        "embedding": embedding,

        "created_at": datetime.now()
    }

    users_collection.insert_one(
        user_data
    )

    print("\n===== REGISTRATION =====")

    print("Name:", name)

    print("User ID:", user_id)

    print("Model:", selected_model)

    print("Embedding Length:", len(embedding))

    print("User stored in MongoDB!")

    if os.path.exists(webm_path):

        os.remove(webm_path)

    if os.path.exists(wav_path):

        os.remove(wav_path)

    return jsonify({

        'status': 'success',

        'name': name,

        'model': selected_model,

        'message': 'Audio registered successfully'
    })

# ======================================================
# VERIFY USER
# ======================================================

@app.route(
    '/verify',
    methods=['POST']
)
def verify_user():

    if users_collection is None:

        return jsonify({

            'status': 'error',

            'message': 'MongoDB server not running'
        })

    if 'audio' not in request.files:

        return jsonify({

            'status': 'error',

            'message': 'No audio received'
        })

    audio = request.files['audio']

    if audio.filename == '':

        return jsonify({

            'status': 'error',

            'message': 'Invalid audio file'
        })

    name = request.form.get('name')

    user_id = request.form.get('user_id')

    selected_model = request.form.get('model')

    if not name or not user_id:

        return jsonify({

            'status': 'error',

            'message': 'Name and User ID required'
        })

    timestamp = datetime.now().strftime(
        '%Y%m%d_%H%M%S'
    )

    safe_user_id = secure_filename(
        user_id
    )

    filename = (
        f'verify_{safe_user_id}_{timestamp}.webm'
    )

    webm_path = os.path.join(
        RECORDINGS_FOLDER,
        filename
    )

    audio.save(webm_path)

    wav_path = convert_webm_to_wav(
        webm_path
    )

    if wav_path is None:

        return jsonify({

            'status': 'error',

            'message': 'Audio conversion failed'
        })

    new_embedding = generate_embedding(

        wav_path,

        selected_model
    )

    if new_embedding is None:

        return jsonify({

            'status': 'error',

            'message': 'Model inference failed'
        })

    stored_user = users_collection.find_one({

        'user_id': user_id,

        'model': selected_model
    })

    if not stored_user:

        return jsonify({

            'status': 'error',

            'message': 'User not found'
        })

    stored_embedding = stored_user['embedding']

    similarity = compute_cosine_similarity(

        stored_embedding,

        new_embedding
    )
    print("\n==========================")
    print("REAL SIMILARITY:", similarity)
    print("==========================\n")
    print("STORED EMBEDDING:")
    print(np.array(stored_embedding)[:10])

    print("\nNEW EMBEDDING:")
    print(np.array(new_embedding)[:10])
    similarity_percentage = round(
        similarity * 100,
        2
    )

    MODEL_THRESHOLDS = {

        "ecapa": 0.90,

        "cta": 0.92,

        "ecapa_conformer": 0.91,

        "ebranchformer": 0.91,
        
        "ecapa_kathbath": 0.91
    }

    threshold = MODEL_THRESHOLDS.get(
        selected_model,
        0.75
    )

    if similarity > threshold:

        result = 'Speaker Verified'

        status = 'success'

    else:

        result = 'Speaker Not Matched'

        status = 'failed'

    print("\n===== VERIFICATION =====")

    print("Name:", name)

    print("User ID:", user_id)

    print("Model:", selected_model)

    print("Similarity:", similarity_percentage)

    print("Result:", result)

    if os.path.exists(webm_path):

        os.remove(webm_path)

    if os.path.exists(wav_path):

        os.remove(wav_path)

    return jsonify({

        'status': status,

        'name': name,

        'message': result,

        'similarity': f'{similarity_percentage}%'
    })

# ======================================================
# MAIN
# ======================================================

if __name__ == '__main__':

    app.run(debug=True)