// Global variables
let audioContext = null;
let analyser = null;
let classifier = null;
let isListening = false;
let isProcessing = false;
let lastDetectionTime = 0;
let animationFrame = null;
let recognition = null;
let classifyInterval = null;
let mediaStream = null;
let audioBuffer = [];
let edgeImpulseModule = null;
let classifierInitialized = false;
let scriptProcessor = null;

// Configuration
let config = {
    threshold: 0.96,
    cooldown: 2000, // milliseconds
    apiUrl: 'https://datinapi.asyncsunlight.tech/query',
    sampleRate: 16000, // Edge Impulse typically uses 16kHz
    frameLength: 16000, // 1 second of audio at 16kHz
};

// DOM Elements
const micButton = document.getElementById('micButton');
const statusText = document.getElementById('status');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const confidenceValue = document.getElementById('confidenceValue');
const settingsButton = document.getElementById('settingsButton');
const settingsPanel = document.getElementById('settingsPanel');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const cooldownSlider = document.getElementById('cooldownSlider');
const cooldownValue = document.getElementById('cooldownValue');
const responsePanel = document.getElementById('responsePanel');
const responseText = document.getElementById('responseText');
const historyPanel = document.getElementById('historyPanel');
const historyList = document.getElementById('historyList');
const waveformContainer = document.getElementById('waveform');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    createWaveformBars();
});

async function initializeApp() {
    updateStatus('Initializing...');
    
    try {
        // Initialize Web Audio API
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: config.sampleRate
        });
        
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        
        // Load Edge Impulse Module
        await loadEdgeImpulseModule();
        
        updateStatus('Ready to start');
    } catch (err) {
        updateStatus('Error initializing: ' + err.message);
        console.error('Initialization error:', err);
    }
}

async function loadEdgeImpulseModule() {
    updateStatus('Loading Edge Impulse model...');
    
    try {
        // Check if Module is available (loaded from edge-impulse-standalone.js)
        if (typeof Module === 'undefined') {
            throw new Error('Edge Impulse module not found. Make sure edge-impulse-standalone.js is loaded before script.js');
        }
        
        edgeImpulseModule = Module;
        
        // Wait for the module to initialize
        if (edgeImpulseModule.calledRun) {
            // Already initialized
            console.log('Module already initialized');
            await initClassifier();
        } else {
            // Wait for initialization
            console.log('Waiting for module initialization...');
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Module initialization timeout (10s)'));
                }, 10000);
                
                edgeImpulseModule.onRuntimeInitialized = async () => {
                    clearTimeout(timeout);
                    console.log('Module runtime initialized');
                    try {
                        await initClassifier();
                        resolve();
                    } catch (err) {
                        reject(err);
                    }
                };
            });
        }
        
        updateStatus('Model loaded successfully');
        console.log('Edge Impulse model ready');
        
    } catch (err) {
        console.error('Model loading error:', err);
        updateStatus('Error loading model: ' + err.message);
        throw err;
    }
}

async function initClassifier() {
    console.log('Initializing classifier...');
    
    // Initialize the classifier
    let ret = edgeImpulseModule.init();
    if (ret !== 0) {
        throw new Error('Classifier init failed with code: ' + ret);
    }
    
    classifierInitialized = true;
    console.log('Classifier initialized successfully');
    
    // Get project info
    try {
        const projectInfo = edgeImpulseModule.get_project();
        if (projectInfo) {
            console.log('Project:', projectInfo.owner + ' / ' + projectInfo.name);
            console.log('Version:', projectInfo.deploy_version);
        }
    } catch (err) {
        console.warn('Could not get project info:', err);
    }
    
    // Get properties
    try {
        const properties = edgeImpulseModule.get_properties();
        if (properties) {
            console.log('Model type:', properties.model_type);
            console.log('Sensor:', properties.sensor);
            config.frameLength = properties.frame_sample_count || 16000;
            config.sampleRate = properties.frequency || 16000;
            console.log('Frame length:', config.frameLength);
            console.log('Sample rate:', config.sampleRate);
        }
    } catch (err) {
        console.warn('Could not get properties:', err);
    }
    
    // Create classifier wrapper
    classifier = {
        classify: async function(rawData) {
            if (!classifierInitialized) {
                throw new Error('Classifier not initialized');
            }
            
            // Allocate memory for the data
            const dataPtr = edgeImpulseModule._malloc(rawData.length * 4);
            const heapFloat32 = new Float32Array(
                edgeImpulseModule.HEAPF32.buffer,
                dataPtr,
                rawData.length
            );
            heapFloat32.set(rawData);
            
            try {
                // Run classifier
                const result = edgeImpulseModule.run_classifier(
                    dataPtr,
                    rawData.length,
                    false // debug
                );
                
                if (result.result !== 0) {
                    throw new Error('Classification failed with code: ' + result.result);
                }
                
                // Parse results
                const output = {
                    anomaly: result.anomaly,
                    results: []
                };
                
                // Extract classification results
                for (let i = 0; i < result.size(); i++) {
                    const item = result.get(i);
                    output.results.push({
                        label: item.label,
                        value: item.value
                    });
                    item.delete();
                }
                
                result.delete();
                return output;
                
            } finally {
                // Free the allocated memory
                edgeImpulseModule._free(dataPtr);
            }
        }
    };
}

function setupEventListeners() {
    // Microphone button
    micButton.addEventListener('click', toggleListening);
    
    // Settings button
    settingsButton.addEventListener('click', () => {
        const isVisible = settingsPanel.style.display === 'block';
        settingsPanel.style.display = isVisible ? 'none' : 'block';
    });
    
    // Threshold slider
    thresholdSlider.addEventListener('input', (e) => {
        config.threshold = parseInt(e.target.value) / 100;
        thresholdValue.textContent = e.target.value;
    });
    
    // Cooldown slider
    cooldownSlider.addEventListener('input', (e) => {
        config.cooldown = parseInt(e.target.value);
        cooldownValue.textContent = (config.cooldown / 1000).toFixed(1);
    });
}

function createWaveformBars() {
    for (let i = 0; i < 50; i++) {
        const bar = document.createElement('div');
        bar.className = 'waveform-bar';
        bar.style.height = '20%';
        waveformContainer.appendChild(bar);
    }
}

async function toggleListening() {
    if (isListening) {
        stopListening();
    } else {
        await startListening();
    }
}

async function startListening() {
    try {
        if (!classifierInitialized) {
            updateStatus('Model not loaded yet. Please wait...');
            return;
        }
        
        // Resume audio context if suspended
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        
        // Get microphone access
        const constraints = {
            audio: {
                sampleRate: config.sampleRate,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        };
        
        console.log('Requesting microphone access...');
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log('Microphone access granted');
        
        // Create audio processing chain
        const source = audioContext.createMediaStreamSource(mediaStream);
        
        // Create script processor for audio capture
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
            if (!isListening) return;
            
            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
            
            // Add to buffer
            audioBuffer.push(...inputData);
            
            // Keep buffer size manageable (store up to 2 seconds)
            const maxBufferSize = config.sampleRate * 2;
            if (audioBuffer.length > maxBufferSize) {
                audioBuffer = audioBuffer.slice(-maxBufferSize);
            }
        };
        
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        source.connect(analyser);
        
        isListening = true;
        micButton.classList.add('listening');
        updateStatus('🎤 Listening for keyword...');
        confidenceDisplay.style.display = 'flex';
        
        console.log('Started listening. Buffer size:', config.frameLength, 'samples');
        
        // Update icon
        const icon = micButton.querySelector('.mic-icon');
        icon.setAttribute('data-lucide', 'mic-off');
        lucide.createIcons();
        
        // Start waveform animation
        updateWaveform();
        
        // Start classification loop
        classifyInterval = setInterval(async () => {
            await classifyAudio();
        }, 500); // Classify every 500ms
        
    } catch (err) {
        updateStatus('Error accessing microphone: ' + err.message);
        console.error('Microphone access error:', err);
        isListening = false;
    }
}

function stopListening() {
    console.log('Stopping listening...');
    isListening = false;
    micButton.classList.remove('listening');
    updateStatus('Stopped');
    confidenceDisplay.style.display = 'none';
    
    // Update icon
    const icon = micButton.querySelector('.mic-icon');
    icon.setAttribute('data-lucide', 'mic');
    lucide.createIcons();
    
    // Stop waveform animation
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
    
    // Clear classification interval
    if (classifyInterval) {
        clearInterval(classifyInterval);
        classifyInterval = null;
    }
    
    // Disconnect and clean up script processor
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor.onaudioprocess = null;
        scriptProcessor = null;
    }
    
    // Stop media stream
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => {
            track.stop();
            console.log('Stopped track:', track.kind);
        });
        mediaStream = null;
    }
    
    // Clear audio buffer
    audioBuffer = [];
    
    // Reset waveform
    const bars = waveformContainer.querySelectorAll('.waveform-bar');
    bars.forEach(bar => {
        bar.classList.remove('active');
        bar.style.height = '20%';
    });
    
    console.log('Listening stopped');
}

function updateWaveform() {
    if (!analyser || !isListening) return;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);
    
    const bars = waveformContainer.querySelectorAll('.waveform-bar');
    const step = Math.floor(bufferLength / bars.length);
    
    bars.forEach((bar, i) => {
        const value = dataArray[i * step] / 128 - 1;
        const height = Math.abs(value) * 80 + 20;
        bar.style.height = `${height}%`;
        bar.classList.add('active');
    });
    
    animationFrame = requestAnimationFrame(updateWaveform);
}

async function classifyAudio() {
    if (!classifierInitialized || !isListening) {
        return;
    }
    
    // Check if we have enough audio data
    if (audioBuffer.length < config.frameLength) {
        console.log(`Waiting for audio buffer: ${audioBuffer.length}/${config.frameLength}`);
        return;
    }
    
    try {
        // Get the last frame worth of audio
        const frame = audioBuffer.slice(-config.frameLength);
        
        // Run classification
        const result = await classifier.classify(frame);
        
        if (!result || !result.results || result.results.length === 0) {
            console.warn('No classification results');
            return;
        }
        
        // Find the highest confidence result
        let maxResult = result.results[0];
        for (const item of result.results) {
            if (item.value > maxResult.value) {
                maxResult = item;
            }
        }
        
        // Log all results for debugging
        console.log('Classification results:', result.results.map(r => `${r.label}: ${(r.value * 100).toFixed(1)}%`).join(', '));
        
        // Update confidence display
        confidenceValue.textContent = `${maxResult.label}: ${(maxResult.value * 100).toFixed(1)}%`;
        
        // Check if detection meets criteria
        const currentTime = Date.now();
        const timeSinceLastDetection = currentTime - lastDetectionTime;
        
        // Check if it's the keyword
        // Adjust these conditions based on your model's output labels
        const isKeyword = maxResult.label.toLowerCase().includes('hello') || 
                          maxResult.label.toLowerCase().includes('airi') ||
                          maxResult.label.toLowerCase() === 'keyword' ||
                          maxResult.label.toLowerCase() === 'hello airi'; // Add your keyword label here
        
        if (maxResult.value >= config.threshold && isKeyword) {
            console.log(`🎯 Keyword detected! Label: ${maxResult.label}, Confidence: ${(maxResult.value * 100).toFixed(1)}%`);
            
            if (timeSinceLastDetection >= config.cooldown) {
                lastDetectionTime = currentTime;
                await handleKeywordDetection(maxResult.label, maxResult.value);
            } else {
                const remaining = ((config.cooldown - timeSinceLastDetection) / 1000).toFixed(1);
                console.log(`⏳ Cooldown active. ${remaining}s remaining`);
            }
        }
    } catch (err) {
        console.error('Classification error:', err);
        updateStatus('Classification error - check console');
    }
}

async function handleKeywordDetection(keyword, confidence) {
    if (isProcessing) return;
    
    isProcessing = true;
    micButton.disabled = true;
    
    // Log detection
    const detection = {
        timestamp: new Date().toISOString(),
        keyword: keyword,
        confidence: confidence
    };
    
    addToHistory(detection);
    
    // Play detection sound (optional)
    playBeep();
    
    // Capture voice query
    const spokenQuery = await captureVoiceQuery();
    
    if (spokenQuery) {
        // Send to API
        await sendQueryToAPI(spokenQuery);
    }
    
    isProcessing = false;
    micButton.disabled = false;
}

function playBeep() {
    if (!audioContext) return;
    
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.2);
}

function captureVoiceQuery() {
    return new Promise((resolve) => {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            updateStatus('Speech recognition not supported');
            resolve(null);
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        updateStatus('🎙️ Listening for your query...');
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            updateStatus(`Heard: "${transcript}"`);
            resolve(transcript);
        };
        
        recognition.onerror = (event) => {
            updateStatus('Error recognizing speech: ' + event.error);
            console.error('Speech recognition error:', event.error);
            resolve(null);
        };
        
        recognition.onend = () => {
            updateStatus('🎤 Listening for keyword...');
        };
        
        recognition.start();
        
        // Timeout after 5 seconds
        setTimeout(() => {
            if (recognition) {
                recognition.stop();
                resolve(null);
            }
        }, 5000);
    });
}

async function sendQueryToAPI(query) {
    try {
        updateStatus('Processing query...');
        
        const response = await fetch(config.apiUrl, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        if (response.ok) {
            const data = await response.json();
            const responseMsg = data.message?.query_resp || 'No response';
            
            // Display response
            displayResponse(responseMsg);
            
            // Speak response
            speakResponse(responseMsg);
            
            updateStatus('🎤 Listening for keyword...');
        } else {
            updateStatus('API request failed');
            console.error('API request failed:', response.status);
        }
    } catch (err) {
        updateStatus('Error querying API: ' + err.message);
        console.error('API query error:', err);
    }
}

function speakResponse(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 1;
        window.speechSynthesis.speak(utterance);
    }
}

function displayResponse(text) {
    responseText.textContent = text;
    responsePanel.style.display = 'block';
    
    // Re-initialize icons for the response panel
    lucide.createIcons();
}

function addToHistory(detection) {
    const item = document.createElement('div');
    item.className = 'history-item';
    
    const time = new Date(detection.timestamp).toLocaleTimeString();
    
    item.innerHTML = `
        <span class="history-time">${time}</span>
        <span class="history-keyword">${detection.keyword}</span>
        <span class="history-confidence">${(detection.confidence * 100).toFixed(1)}%</span>
    `;
    
    historyList.insertBefore(item, historyList.firstChild);
    historyPanel.style.display = 'block';
    
    // Keep only last 10 detections
    if (historyList.children.length > 10) {
        historyList.removeChild(historyList.lastChild);
    }
}

function updateStatus(message) {
    statusText.textContent = message;
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isListening) {
        stopListening();
    }
    if (audioContext) {
        audioContext.close();
    }
});