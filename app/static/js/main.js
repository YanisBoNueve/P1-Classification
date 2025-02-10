document.addEventListener('DOMContentLoaded', function() {
    // Show content after initial load
    setTimeout(() => {
        document.getElementById('loading-screen').style.display = 'none';
        document.querySelector('.container').style.display = 'block';
        setTimeout(() => {
            document.querySelector('.container').classList.add('loaded');
        }, 50);
    }, 1000);

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.querySelector('.upload-button');
    const resultsSection = document.querySelector('.results-section');
    const analysisLoading = document.getElementById('analysis-loading');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    const loadingStatus = document.getElementById('loading-status');
    const logContainer = document.querySelector('.log-container');


    function addLog(message) {
        const time = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    function checkInitStatus() {
        fetch('/init-status')
            .then(response => response.json())
            .then(data => {
                progressFill.style.width = `${data.progress}%`;
                progressText.textContent = `${data.progress}%`;
                loadingStatus.textContent = data.status;
                addLog(data.status);

                if (data.progress < 100) {
                    setTimeout(checkInitStatus, 500);
                } else {
                    setTimeout(() => {
                        document.getElementById('loading-screen').style.display = 'none';
                        document.querySelector('.container').style.display = 'block';
                        document.querySelector('.container').classList.add('loaded');
                    }, 1000);
                }
            })
            .catch(error => {
                addLog(`Error: ${error.message}`);
                setTimeout(checkInitStatus, 1000);
            });
    }

    // Start checking initialization status
    addLog("Starting initialization...");
    checkInitStatus();

    // Handle file upload
    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.name.endsWith('.dcm')) {
            alert('Please upload a DICOM file (.dcm)');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Show loading
        analysisLoading.style.display = 'flex';
        resultsSection.style.display = 'none';

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            alert('Error: ' + error.message);
        })
        .finally(() => {
            analysisLoading.style.display = 'none';
        });
    }

    // Event listeners
    uploadButton.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload({ target: fileInput });
        }
    });

    function displayResults(data) {
        // On s'assure que la section des résultats est visible
        resultsSection.style.display = 'block';
        
        // On vide le contenu précédent (s'il y en a)
        resultsSection.innerHTML = '';
    
        // Si le back-end a retourné une URL pour l'image d'analyse
        if (data.analysis_image) {
            const imgElem = document.createElement('img');
            imgElem.src = data.analysis_image; // L'URL renvoyée par le back-end
            imgElem.alt = 'Résultat de l\'analyse';
            imgElem.style.maxWidth = '100%';  // Pour s'adapter à la taille du conteneur
            resultsSection.appendChild(imgElem);
        }
    
        // Affichage des informations complémentaires
        const infoElem = document.createElement('div');
        infoElem.className = 'analysis-info';
        infoElem.innerHTML = `
            <p><strong>Prediction :</strong> ${data.predicted_class}</p>
            <p><strong>Confiance :</strong> ${(data.confidence * 100).toFixed(2)}%</p>
            <p><strong>Temps de traitement :</strong> ${data.processing_time.toFixed(2)} secondes</p>
        `;
        resultsSection.appendChild(infoElem);
    }
    
});