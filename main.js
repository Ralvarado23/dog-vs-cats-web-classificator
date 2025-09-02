let model = null;
let currentImageBitmap = null;

const elements = {
    modelStatus: document.getElementById('modelStatus'),
    statusText: document.getElementById('statusText'),
    uploadSection: document.getElementById('uploadSection'),
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    predictionSection: document.getElementById('predictionSection'),
    imagePreview: document.getElementById('imagePreview'),
    predictBtn: document.getElementById('predictBtn'),
    clearBtn: document.getElementById('clearBtn'),
    resultsContainer: document.getElementById('resultsContainer'),
    dogResult: document.getElementById('dogResult'),
    catResult: document.getElementById('catResult'),
    dogPercentage: document.getElementById('dogPercentage'),
    catPercentage: document.getElementById('catPercentage'),
    confidenceNote: document.getElementById('confidenceNote'),
    workCanvas: document.getElementById('workCanvas')
};

function setModelStatus(type, message) {
    const dot = elements.modelStatus.querySelector('.status-dot');
    elements.modelStatus.className = `model-status ${type}`;
    dot.className = `status-dot ${type}`;
    elements.statusText.textContent = message;
}

async function loadModel() {
    try {
    setModelStatus('loading', 'Cargando modelo...');

    try {
        model = await tf.loadLayersModel('modelo/model.json');
    } catch {
        model = await tf.loadGraphModel('modelo/model.json');
    }

    setModelStatus('ready', 'Modelo listo');
    updatePredictButton();

    } catch (error) {
    console.error('Error cargando modelo:', error);
    setModelStatus('error', 'Error al cargar modelo');
    }
}

function updatePredictButton() {
    elements.predictBtn.disabled = !(model && currentImageBitmap);
}

function getModelInputSize() {
    try {
    const shape = model?.inputs?.[0]?.shape;
    const height = shape?.[1] || 120;
    const width = shape?.[2] || 120;
    return { width, height };
    } catch {
    return { width: 120, height: 120 };
    }
}

async function preprocessImage(imageBitmap) {
    const { width, height } = getModelInputSize();

    elements.workCanvas.width = width;
    elements.workCanvas.height = height;

    const ctx = elements.workCanvas.getContext('2d');

    // Calcular crop centrado cuadrado
    const size = Math.min(imageBitmap.width, imageBitmap.height);
    const offsetX = (imageBitmap.width - size) / 2;
    const offsetY = (imageBitmap.height - size) / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(imageBitmap, offsetX, offsetY, size, size, 0, 0, width, height);

    return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(elements.workCanvas);
    return tensor.toFloat().expandDims(0);
    });
}

function interpretPrediction(output) {
    const data = output.dataSync();

    if (data.length === 1) {
    // Salida binaria: 0=gato, 1=perro
    const dogProb = data[0];
    const catProb = 1 - dogProb;
    return { dogProb, catProb };
    } else if (data.length === 2) {
    // Dos salidas: [gato, 1-perro]
    return { catProb: data[0], dogProb: data[1] };
    } else {
    // Más salidas, asumir [gato, perro, ...]
    return { catProb: data[0] || 0, dogProb: data[1] || 0 };
    }
}

// Función para animar el conteo de porcentajes
function animatePercentage(element, targetValue, duration = 1500) {
    let startValue = 0;
    const increment = targetValue / (duration / 16); // 60fps aproximadamente
    let currentValue = startValue;

    const animate = () => {
    currentValue += increment;
    if (currentValue >= targetValue) {
        currentValue = targetValue;
        element.textContent = `${Math.round(currentValue)}%`;
        element.classList.add('counting');
        setTimeout(() => element.classList.remove('counting'), 300);
        return;
    }
    element.textContent = `${Math.round(currentValue)}%`;
    requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
}

async function predict() {
    if (!model || !currentImageBitmap) return;

    // Deshabilitar el botón y añadir clase a controls
    const predictBtn = elements.predictBtn;
    const controls = document.querySelector('.controls');
    predictBtn.disabled = true;
    controls.classList.add('predicting');

    // Añadir estado de loading al botón
    const originalContent = predictBtn.innerHTML;
    predictBtn.innerHTML = '<div class="loading-spinner"></div><span>Analizando...</span>';
    predictBtn.classList.add('predicting');
    
    elements.resultsContainer.classList.remove('show');

    try {
    const inputTensor = await preprocessImage(currentImageBitmap);

    let prediction = model.predict(inputTensor);
    const output = Array.isArray(prediction) ? prediction[0] : prediction;

    const { dogProb, catProb } = interpretPrediction(output);

    // Calcular porcentajes
    const dogPercent = Math.round(dogProb * 100);
    const catPercent = Math.round(catProb * 100);

    // Resetear estado visual
    elements.dogResult.classList.remove('winner', 'animate-in');
    elements.catResult.classList.remove('winner', 'animate-in');
    elements.dogPercentage.textContent = '0%';
    elements.catPercentage.textContent = '0%';
    elements.confidenceNote.classList.remove('show');

    // Mostrar contenedor con delay
    setTimeout(() => {
        elements.resultsContainer.classList.add('show');
        
        // Animar entrada de elementos
        setTimeout(() => {
        elements.dogResult.classList.add('animate-in');
        elements.catResult.classList.add('animate-in');
        }, 200);

        // Animar porcentajes con delay escalonado
        setTimeout(() => {
        animatePercentage(elements.dogPercentage, dogPercent, 1200);
        }, 600);

        setTimeout(() => {
        animatePercentage(elements.catPercentage, catPercent, 1200);
        }, 800);

        // Determinar y animar ganador
        setTimeout(() => {
        if (dogProb > catProb) {
            elements.dogResult.classList.add('winner');
        } else {
            elements.catResult.classList.add('winner');
        }

        // Mostrar nota de confianza
        const maxProb = Math.max(dogProb, catProb);
        let confidenceText = '';
        if (maxProb > 0.9) {
            confidenceText = '🎯 Confianza Muy Alta - Predicción muy confiable';
        } else if (maxProb > 0.7) {
            confidenceText = '✅ Confianza Alta - Predicción confiable';
        } else if (maxProb > 0.6) {
            confidenceText = '🤔 Confianza Media - Predicción moderada';
        } else {
            confidenceText = '❓ Confianza Baja - Predicción poco confiable';
        }
        elements.confidenceNote.textContent = confidenceText;
        
        setTimeout(() => {
            elements.confidenceNote.classList.add('show');
        }, 200);

        }, 1400);

    }, 300);

    // Limpiar memoria
    inputTensor.dispose();
    if (Array.isArray(prediction)) {
        prediction.forEach(p => p.dispose());
    } else {
        prediction.dispose();
    }

    } catch (error) {
    console.error('Error en predicción:', error);
    alert('Error realizando predicción: ' + error.message);
    // En caso de error, mantener el botón deshabilitado
    } finally {
    // Restaurar apariencia del botón pero mantenerlo deshabilitado
    setTimeout(() => {
        predictBtn.innerHTML = originalContent;
        predictBtn.classList.remove('predicting');
    }, 1800);
    }
}

function handleImageLoad(file) {
    if (!file?.type.startsWith('image/')) {
    alert('Por favor selecciona un archivo de imagen válido');
    return;
    }

    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = async () => {
    try {
        currentImageBitmap = await createImageBitmap(img);
        
        // Iniciar transición de salida de upload
        elements.uploadSection.classList.add('hiding');
        
        // Configurar imagen
        elements.imagePreview.src = url;
        elements.imagePreview.classList.remove('loaded');
        
        // Después de que termine la transición de salida
        setTimeout(() => {
        elements.predictionSection.classList.add('show');
        
        // Animar entrada de imagen después de un pequeño delay
        setTimeout(() => {
            elements.imagePreview.classList.add('loaded');
        }, 200);
        
        updatePredictButton();
        }, 600);

    } catch (error) {
        console.error('Error procesando imagen:', error);
        alert('No se pudo procesar la imagen');
    }
    };

    img.onerror = () => {
    alert('Error cargando imagen');
    URL.revokeObjectURL(url);
    };

    img.src = url;
}

function clearImage() {
    currentImageBitmap = null;
    elements.imagePreview.src = '';
    elements.fileInput.value = '';

    // Animar salida de prediction section
    elements.predictionSection.classList.remove('show');
    elements.resultsContainer.classList.remove('show');
    
    setTimeout(() => {
    elements.uploadSection.classList.remove('hiding');
    
    // Limpiar resultados
    elements.dogResult.classList.remove('winner', 'animate-in');
    elements.catResult.classList.remove('winner', 'animate-in');
    elements.dogPercentage.textContent = '0%';
    elements.catPercentage.textContent = '0%';
    elements.confidenceNote.classList.remove('show');
    elements.imagePreview.classList.remove('loaded');
    
    updatePredictButton();
    }, 300);

    // Restaurar el margen de controls
    const controls = document.querySelector('.controls');
    controls.classList.remove('predicting');
}

// Event listeners
elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleImageLoad(e.target.files[0]);
});

elements.uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
});

elements.uploadArea.addEventListener('dragleave', () => {
    elements.uploadArea.classList.remove('dragover');
});

elements.uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files?.[0];
    if (file) handleImageLoad(file);
});

elements.predictBtn.addEventListener('click', predict);
elements.clearBtn.addEventListener('click', clearImage);

// Inicializar
loadModel();