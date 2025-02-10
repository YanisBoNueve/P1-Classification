import os
import uuid
import numpy as np
import pydicom
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tensorflow.keras.models import load_model
from matplotlib.patches import Rectangle
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DicomSeriesLoader:
    """
    Handles loading of DICOM files from nested directories
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.series_dict = defaultdict(list)
        
    def scan_for_dicoms(self):
        """Recursively scans directories for DICOM files"""
        logger.info(f"Scanning for DICOM files in {self.base_dir}")
        total_files = 0
        
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.dcm'):
                    file_path = os.path.join(root, file)
                    try:
                        # Read basic DICOM metadata without loading pixel data
                        dicom = pydicom.dcmread(file_path, stop_before_pixels=True)
                        series_id = getattr(dicom, 'SeriesDescription', os.path.basename(root))
                        self.series_dict[series_id].append(file_path)
                        total_files += 1
                    except Exception as e:
                        logger.warning(f"Error reading DICOM header for {file}: {e}")
        
        logger.info(f"Found {total_files} DICOM files in {len(self.series_dict)} series")
        return self.series_dict
    
    def get_series_info(self):
        """Returns information about found series"""
        return {
            series: len(files) 
            for series, files in self.series_dict.items()
        }

class EnhancedVisualizer:
    """
    Handles advanced visualization of mammogram analysis results
    """
    def __init__(self, class_names):
        self.class_names = class_names
        self.colors = sns.color_palette("husl", len(class_names))
    

    def create_detailed_visualization(self, img, results, save_path=None):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 10))
        
        # Création des sous-graphes
        self._plot_mammogram(plt.subplot2grid((2, 4), (0, 0), colspan=2), img, results)
        self._plot_probabilities(plt.subplot2grid((2, 4), (0, 2), colspan=2), results)
        self._plot_confidence_gauge(plt.subplot2grid((2, 4), (1, 0)), results['confidence'])
        self._plot_metrics_summary(plt.subplot2grid((2, 4), (1, 1)), results)
        self._plot_dicom_info(plt.subplot2grid((2, 4), (1, 2), colspan=2), results)
        
        plt.tight_layout()
        
        if not save_path:
            unique_name = f"{uuid.uuid4()}.png"
            # Calcule le chemin absolu pour le dossier static "analysis"
            # __file__ est le chemin vers ce fichier (dans app/analyzer)
            # On remonte d'un niveau pour arriver dans le dossier "app"
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Construit le chemin vers le dossier "static/analysis" à l'intérieur de "app"
            save_dir = os.path.join(base_dir, "static", "analysis")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, unique_name)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Retourne l'URL accessible (le dossier statique est servi depuis "app/static")
        return "/static/analysis/" + os.path.basename(save_path)


    
    def _plot_mammogram(self, ax, img, results):
        ax.imshow(img, cmap='gray')
        ax.set_title('Processed Mammogram')
        ax.axis('off')
        
        confidence = results['confidence']
        color = 'green' if confidence > 0.8 else 'yellow' if confidence > 0.6 else 'red'
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                        facecolor='none', edgecolor=color, linewidth=3)
        ax.add_patch(rect)
    
    def _plot_probabilities(self, ax, results):
        probabilities = list(results['probabilities'].values())
        bars = ax.bar(self.class_names, probabilities, color=self.colors)
        ax.set_title('Prediction Probabilities')
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
    
    def _plot_confidence_gauge(self, ax, confidence):
        ax.set_title('Confidence Level')
        
        colors = ['red', 'yellow', 'green']
        n_colors = len(colors)
        
        for i in range(n_colors):
            ax.barh(0, 1/n_colors, left=i/n_colors, color=colors[i], alpha=0.3)
        
        ax.barh(0, 0.02, left=confidence, color='black')
        ax.text(0.5, -0.5, f'{confidence:.1%}', ha='center', va='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')
    
    def _plot_metrics_summary(self, ax, results):
        ax.text(0.5, 0.8, f"Prediction:\n{results['predicted_class']}", 
                ha='center', va='center', fontsize=10)
        ax.text(0.5, 0.2, f"Confidence:\n{results['confidence']:.1%}", 
                ha='center', va='center', fontsize=10)
        ax.axis('off')
    
    def _plot_dicom_info(self, ax, results):
        ax.text(0.05, 0.95, "DICOM Information:", fontsize=10, va='top')
        y_pos = 0.8
        
        dicom_info = results.get('dicom_metadata', {})
        important_fields = [
            'SeriesDescription', 'StudyDate', 'PatientID',
            'Modality', 'ImageLaterality', 'ViewPosition'
        ]
        
        for field in important_fields:
            value = dicom_info.get(field, 'Unknown')
            ax.text(0.1, y_pos, f"{field}: {value}", fontsize=8)
            y_pos -= 0.15
        
        ax.axis('off')

class MammogramAnalyzer:
    """
    Main class for mammogram analysis
    """
    def __init__(self, model_path, img_size=224):
        self.img_size = img_size
        self.model = self.load_model(model_path)
        self.class_names = ['BENIGN_WITHOUT_CALLBACK', 'BENIGN', 'MALIGNANT']
        self.visualizer = EnhancedVisualizer(self.class_names)
    
    def load_model(self, model_path):
        """Loads the trained model"""
        try:
            logger.info(f"Loading model from {model_path}")
            # Load model without compilation
            model = load_model(model_path, compile=False)
            
            # Recompile with legacy optimizer
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def enhance_mammogram(self, img):
        """Enhances mammogram image quality"""
        try:
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip(img, p1, p99)
            img = ((img - p1) / (p99 - p1) * 255).astype(np.uint8)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            return img
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return None

    def preprocess_dicom(self, file_path):
        """Preprocesses DICOM image for analysis"""
        try:
            dicom = pydicom.dcmread(file_path)
            img = dicom.pixel_array.astype(np.float32)
            
            if img.shape[0] < 100 or img.shape[1] < 100:
                raise ValueError("Image dimensions too small")
            
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = self.enhance_mammogram(img)
            img = (img.astype(np.float32) - 127.5) / 127.5
            img = np.stack([img] * 3, axis=-1)
            
            return img, dicom
        except Exception as e:
            logger.error(f"Error preprocessing DICOM {file_path}: {e}")
            return None, None

    def analyze_image(self, file_path, visualization=True, save_path=None):
        try:
            import time
            start_time = time.time()
            
            img, dicom = self.preprocess_dicom(file_path)
            if img is None:
                return None
            
            prediction = self.model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            predicted_class = self.class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            processing_time = time.time() - start_time
            
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, prediction)
                },
                'dicom_metadata': {
                    'SeriesDescription': getattr(dicom, 'SeriesDescription', 'Unknown'),
                    'StudyDate': getattr(dicom, 'StudyDate', 'Unknown'),
                    'PatientID': getattr(dicom, 'PatientID', 'Unknown'),
                    'Modality': getattr(dicom, 'Modality', 'Unknown'),
                    'ImageLaterality': getattr(dicom, 'ImageLaterality', 'Unknown'),
                    'ViewPosition': getattr(dicom, 'ViewPosition', 'Unknown')
                },
                'processing_time': processing_time
            }
            
            if visualization:
                # La méthode va renvoyer l'URL de l'image enregistrée
                image_url = self.visualizer.create_detailed_visualization(img, results, save_path)
                results['analysis_image'] = image_url
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None


def analyze_series(analyzer, series_dict, output_dir='analysis_results'):
    """Analyzes all images in each series"""
    os.makedirs(output_dir, exist_ok=True)
    results_by_series = defaultdict(list)
    
    for series_name, files in series_dict.items():
        logger.info(f"\nAnalyzing series: {series_name}")
        series_dir = os.path.join(output_dir, series_name.replace(" ", "_"))
        os.makedirs(series_dir, exist_ok=True)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            logger.info(f"Processing {filename}")
            
            result = analyzer.analyze_image(
                file_path,
                visualization=True,
                save_path=os.path.join(series_dir, f'analysis_{filename}.png')
            )
            
            if result:
                result['filename'] = filename
                results_by_series[series_name].append(result)
                
                logger.info(f"Prediction: {result['predicted_class']}")
                logger.info(f"Confidence: {result['confidence']:.2%}")
    
    return results_by_series

def main():
    """Main execution"""
    MODEL_PATH = '/Users/dtilm/Desktop/P1-Classification/scripts/final_model.keras'
    BASE_DIR = '/Users/dtilm/Desktop/P1-Classification/test_images/Case1 [Case1]/20080408 023126 [ - BREAST IMAGING TOMOSYNTHESIS]/'  # Update this path
    
    try:
        # Initialize DICOM loader and scan for files
        loader = DicomSeriesLoader(BASE_DIR)
        series_dict = loader.scan_for_dicoms()
        
        if not series_dict:
            logger.error("No DICOM files found")
            return
        
        # Print series information
        logger.info("\nFound series:")
        for series, count in loader.get_series_info().items():
            logger.info(f"{series}: {count} files")
        
        # Initialize analyzer
        analyzer = MammogramAnalyzer(MODEL_PATH)
        
        # Analyze all series
        results = analyze_series(analyzer, series_dict)
        
        # Print summary statistics
        logger.info("\nAnalysis Summary:")
        for series_name, series_results in results.items():
            confidence_levels = [r['confidence'] for r in series_results]
            logger.info(f"\nSeries: {series_name}")
            logger.info(f"Total images: {len(series_results)}")
            logger.info(f"Average confidence: {np.mean(confidence_levels):.2%}")
            logger.info("Predictions by class:")
            for class_name in analyzer.class_names:
                count = sum(1 for r in series_results if r['predicted_class'] == class_name)
                logger.info(f"  {class_name}: {count}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()