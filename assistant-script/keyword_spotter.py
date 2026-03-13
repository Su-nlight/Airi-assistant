import os
import sys
import signal
import time
import logging
from datetime import datetime
from edge_impulse_linux.audio import AudioImpulseRunner
import speech_recognition as sr
import requests
import pyttsx3

# Configuration from environment variables
MODEL_PATH = os.getenv('MODEL_PATH', './datin-voice-agent-mac-arm64-v14.eim')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', 0.96))
COOLDOWN_SECONDS = float(os.getenv('COOLDOWN_SECONDS', 2.0))
LOG_FILE = './logs/detections.log'
CSV_FILE = './logs/detections.csv'

# Setup logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global runner for signal handler
runner = None
last_detection_time = 0


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info('Received shutdown signal')
    if runner:
        runner.stop()
    sys.exit(0)


def initialize_csv():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w') as f:
            f.write("timestamp,keyword,confidence\n")


def log_detection(keyword, confidence):
    """Log detection to CSV file"""
    timestamp = datetime.now().isoformat()
    
    logger.info("🔔" * 25)
    logger.info(f"   DETECTED: '{keyword}' ({confidence:.2%})")
    logger.info(f"   Time: {timestamp}")
    logger.info("🔔" * 25)
    
    with open(CSV_FILE, 'a') as f:
        f.write(f"{timestamp},{keyword},{confidence:.4f}\n")


def should_trigger_detection(confidence, label, current_time):
    """Check if detection meets criteria for triggering"""
    global last_detection_time
    
    # Check threshold
    if confidence < DETECTION_THRESHOLD:
        return False
    
    # Filter out noise/unknown labels
    if label.lower() in ['noise', 'unknown']:
        return False
    
    # Check cooldown period
    if current_time - last_detection_time < COOLDOWN_SECONDS:
        return False
    
    return True

def keyword_trigger() -> str:
    r = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Say something!")
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        # Listen for audio
        audio = r.listen(source)
    try:
        # Recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
        # pass this text as query to rag agent
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def main():
    global runner, last_detection_time
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        logger.error("Please provide a valid model path via MODEL_PATH environment variable")
        sys.exit(1)
    
    # Initialize CSV
    initialize_csv()
    
    logger.info("=" * 60)
    logger.info("  Edge Impulse Keyword Spotter")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Threshold: {DETECTION_THRESHOLD}")
    logger.info(f"Cooldown: {COOLDOWN_SECONDS}s")
    logger.info("=" * 60)
    
    try:
        with AudioImpulseRunner(MODEL_PATH) as runner:
            # Initialize model
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']
            
            logger.info(f"✓ Loaded: {model_info['project']['owner']} / {model_info['project']['name']}")
            logger.info(f"✓ Labels: {', '.join(labels)}")
            logger.info("=" * 60)
            logger.info("🎤 LISTENING... Press Ctrl+C to stop")
            logger.info("=" * 60)
            
            # Process audio stream
            for res, audio in runner.classifier():
                if "classification" in res["result"]:
                    predictions = res["result"]["classification"]
                    
                    # Find highest confidence prediction
                    max_label = max(predictions, key=predictions.get)
                    max_confidence = predictions[max_label]
                    
                    # Check if detection should trigger
                    current_time = time.time()
                    if should_trigger_detection(max_confidence, max_label, current_time):
                        last_detection_time = current_time
                        log_detection(max_label, max_confidence)
                        query = {"query": keyword_trigger()}
                        response = requests.post("https://datinapi.asyncsunlight.tech/query", params=query, headers={"accept": "application/json"})
                        if response.status_code == 200:
                            data = response.json()
                            print(f"Response for query : {query} \n --::-- \n {data['message']['query_resp']}")
                            engine = pyttsx3.init()
                            # engine.setProperty('voice', 'com.apple.voice.Tara')
                            engine.say(data['message']['query_resp'])
                            engine.runAndWait()
                    
                    # Optional: Print all scores (comment out if too verbose)
                    # logger.debug(f"Result ({res['timing']['dsp'] + res['timing']['classification']}ms): " +
                    #             ", ".join(f"{label}: {score:.2%}" for label, score in predictions.items()))
                
                elif "freeform" in res["result"]:
                    logger.info(f"Freeform result ({res['timing']['dsp'] + res['timing']['classification']}ms)")
                    for i, output in enumerate(res["result"]["freeform"]):
                        logger.info(f"  Output {i}: {', '.join(f'{x:.4f}' for x in output)}")
    
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if runner:
            runner.stop()
        logger.info("✓ Stopped")


if __name__ == "__main__":
    main()