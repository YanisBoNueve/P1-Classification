import matplotlib
matplotlib.use('Agg')
import threading
import time
import os
from flask import Flask, render_template, request, jsonify
from app.analyzer.mammogram_analyzer import MammogramAnalyzer

# Précision des dossiers pour les templates et fichiers statiques
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Affichage des chemins dans la console (optionnel, pour vérification)
print("Chemin des templates:", app.template_folder)
print("Chemin des static:", app.static_folder)

MODEL_PATH = '/Users/dtilm/Desktop/P1-Classification/scripts/final_model.keras'
global_analyzer = None
initialization_progress = 0
initialization_status = "Starting..."

def initialize_analyzer():
    global initialization_progress, initialization_status, global_analyzer

    initialization_status = "Loading TensorFlow..."
    initialization_progress = 10
    time.sleep(1)  # Simuler le chargement

    initialization_status = "Initializing model architecture..."
    initialization_progress = 30
    time.sleep(1)

    initialization_status = "Loading model weights..."
    initialization_progress = 50
    global_analyzer = MammogramAnalyzer(MODEL_PATH)

    initialization_status = "Compiling model..."
    initialization_progress = 80
    time.sleep(1)

    initialization_status = "Ready!"
    initialization_progress = 100

def initialize_analyzer_async():
    thread = threading.Thread(target=initialize_analyzer)
    thread.daemon = True  # Permet au thread de s'arrêter avec le processus principal
    thread.start()

def get_analyzer():
    global global_analyzer
    if global_analyzer is None:
        raise Exception("Le modèle n'est pas encore chargé")
    return global_analyzer

@app.route('/init-status')
def init_status():
    return jsonify({
        'progress': initialization_progress,
        'status': initialization_status
    })

@app.route('/')
def index():
    # Flask cherchera index.html dans le dossier "app/templates"
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        analyzer = get_analyzer()  # Utilise l'instance existante
    except Exception as e:
        return jsonify({'error': str(e)})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not file.filename.endswith('.dcm'):
        return jsonify({'error': 'Please upload a DICOM file'})

    # Sauvegarde le fichier temporairement
    temp_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(temp_path)

    try:
        result = analyzer.analyze_image(temp_path)
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Démarrer le chargement du modèle en arrière-plan
    initialize_analyzer_async()
    app.run(debug=True)
