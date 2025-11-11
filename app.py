import cv2
import face_recognition
import os
import numpy as np
import ollama
import base64
import threading
import time
from retinaface import RetinaFace
from flask import Flask, render_template, Response, jsonify, request

# --- Configuration (Unchanged) ---
MODEL_GEMMA = 'gemma3:4b'
MODEL_LLAVA = 'llava'
MODEL_MOONDREAM = 'moondream'
MODEL_OFF = 'off'
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
ANALYSIS_COOLDOWN = 10
BOX_PADDING = 15
CONFIDENCE_THRESHOLD = 0.6
DETECTOR_CONFIDENCE = 0.9 # For RetinaFace
ACTION_MOTION_THRESHOLD = 30
# ---------------------

# --- Global Server State (Unchanged) ---
data_lock = threading.Lock()
output_frame = None
server_data = {
    "is_recording": False,
    "keyframe_count": 0,
    "action_result": "",
    "live_faces": [],
    "model": MODEL_GEMMA,
    "detection_mode": "accurate", 
    "nth_frame": 2,
    "hog_scale_factor": 0.5
}

# --- Global ML State (Unchanged) ---
known_face_encodings = []
known_face_names = []
analysis_results = {}
action_thread = None
stop_action_thread = False

# --- Flask App Initialization (Unchanged) ---
app = Flask(__name__)

# --- Frame Conversion (Unchanged) ---
def frame_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    if not success: return None
    return base64.b64encode(buffer).decode('utf-8')

# --- Live Action Comprehension Thread (Unchanged) ---
def action_comprehension_thread():
    global data_lock, output_frame, server_data, stop_action_thread
    print("[Action Thread] Started.")
    chat_messages = []
    last_frame_gray = None
    keyframe_count = 0

    while not stop_action_thread:
        current_frame = None
        with data_lock:
            if output_frame is not None:
                current_frame = output_frame.copy()
            current_model = server_data['model'] # Get model for prompt
            
        if current_frame is None:
            time.sleep(0.1); continue
        
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        is_keyframe = False
        if last_frame_gray is None:
            is_keyframe = True
        else:
            frame_delta = cv2.absdiff(last_action_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, ACTION_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            if thresh.sum() > 0: is_keyframe = True
        
        if is_keyframe:
            print(f"[Action Thread] Motion detected! Processing Keyframe {keyframe_count + 1}.")
            last_frame_gray = gray
            keyframe_count += 1
            
            with data_lock: server_data["keyframe_count"] = keyframe_count
            b64_frame = frame_to_base64(current_frame)
            if not b64_frame: continue

            prompt = ""
            if current_model == MODEL_MOONDREAM:
                prompt = "Describe this image."
            else:
                if not chat_messages:
                    prompt = "This is the first keyframe. Briefly describe what is happening."
                else:
                    prompt = "This is the next keyframe. Briefly describe the new action."
            
            chat_messages.append({"role": "user", "content": prompt, "images": [b64_frame]})
            
            try:
                print(f"[Action Thread] Sending to {current_model}...")
                response = ollama.chat(model=current_model, messages=chat_messages, stream=False)
                
                response_message = response['message']
                response_content = response_message['content']
                chat_messages.append(response_message)
                
                with data_lock:
                    server_data["action_result"] += f"- {response_content}\n"
            except Exception as e:
                print(f"[Action Thread] Error calling Ollama: {e}")
                with data_lock: server_data["action_result"] = "Error connecting to Ollama."
                time.sleep(2)
        
        time.sleep(0.5) 
    print("[Action Thread] Stopped.")


# --- Single Frame Analysis (Unchanged) ---
def analyze_frame_with_gemma(frame, name):
    global analysis_results, data_lock
    b64_image = frame_to_base64(frame)
    if b64_image is None:
        with data_lock: analysis_results[name] = "Error: Failed to encode frame."
        return

    with data_lock: current_model = server_data['model']
    if current_model == MODEL_OFF:
        print("[Single Frame] Analysis skipped, model is OFF.")
        return

    prompt = ""
    if current_model == MODEL_MOONDREAM:
        prompt = "Describe the person in this image and what they are doing."
    else:
        prompt = f"This person has been identified as {name}. Briefly describe what they are doing."

    try:
        print(f"[Single Frame] Sending to {current_model}...")
        response = ollama.chat(model=current_model, messages=[{'role': 'user', 'content': prompt, 'images': [b64_image]}], stream=False)
        analysis_text = response['message']['content']
        print(f"\n--- Analysis for {name} ---\n{analysis_text}\n--------------------")
        with data_lock: analysis_results[name] = analysis_text
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        with data_lock: analysis_results[name] = "Error connecting to Ollama."

# --- Face Loading (Unchanged) ---
def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names
    print(f"Loading known faces from {known_faces_dir}...")
    if not os.path.exists(known_faces_dir): return
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir) or person_name.startswith('.'): continue
        print(f"Loading faces for: {person_name}")
        image_count = 0
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(person_dir, filename)
                try:
                    face_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(face_image)
                    for encoding in face_encodings:
                        known_face_encodings.append(encoding)
                        known_face_names.append(person_name)
                    image_count += 1
                except Exception as e:
                    print(f" - ERROR loading {image_path}: {e}")
        if image_count > 0:
            print(f" - Loaded {image_count} images for {person_name}.")

# --- Video Processing Thread (Unchanged) ---
def video_processing_thread():
    global data_lock, output_frame, server_data
    global analysis_results
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    frame_count = 0
    last_analysis_time = {} 
    current_face_data = [] 
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Retrying...")
            time.sleep(1.0)
            continue
            
        frame = cv2.flip(frame, 1)
        frame_count += 1

        with data_lock:
            is_recording = server_data["is_recording"]
            current_model = server_data["model"]
            current_detector = server_data["detection_mode"]
            current_nth_frame = server_data["nth_frame"]
            current_scale_factor = server_data["hog_scale_factor"]

        if is_recording:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            global last_action_frame_gray, action_frames
            if last_action_frame_gray is None:
                last_action_frame_gray = gray
                b64_frame = frame_to_base64(frame)
                if b64_frame: action_frames.append(b64_frame); print(f"[Action Analysis] Saved Keyframe 1 (Start)")
            else:
                frame_delta = cv2.absdiff(last_action_frame_gray, gray)
                thresh = cv2.threshold(frame_delta, ACTION_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                if thresh.sum() > 0:
                    b64_frame = frame_to_base64(frame)
                    if b64_frame: action_frames.append(b64_frame); print(f"[Action Analysis] Saved Keyframe {len(action_frames)} (Motion Detected)")
                    last_action_frame_gray = gray
            with data_lock: server_data["keyframe_count"] = len(action_frames)

        if frame_count % current_nth_frame == 0:
            current_face_data = [] 
            live_face_payload = [] 
            
            face_locations = []
            face_encodings = []
            
            if current_detector == "fast":
                draw_scale_factor = 1 / current_scale_factor
                small_frame = cv2.resize(frame, (0, 0), fx=current_scale_factor, fy=current_scale_factor)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations_small = face_recognition.face_locations(rgb_small_frame, model='hog')
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_small)
                face_locations = [(int(t*draw_scale_factor), int(r*draw_scale_factor), int(b*draw_scale_factor), int(l*draw_scale_factor)) for (t,r,b,l) in face_locations_small]
                
            else: # "accurate"
                try:
                    faces = RetinaFace.detect_faces(frame, threshold=DETECTOR_CONFIDENCE)
                except Exception as e: faces = {}
                
                if isinstance(faces, dict):
                    boxes_trbl = []
                    for face_key, face_data in faces.items():
                        x1, y1, x2, y2 = face_data['facial_area']
                        top, right, bottom, left = int(y1), int(x2), int(y2), int(x1)
                        boxes_trbl.append((top, right, bottom, left))
                    
                    if boxes_trbl:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = boxes_trbl
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                name, confidence = "Unknown", 0
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    if min_distance < CONFIDENCE_THRESHOLD:
                        name = known_face_names[best_match_index]
                    confidence = max(0, min(100, (1.0 - min_distance) * 100))
                
                current_time = time.time()
                
                if (current_model != MODEL_OFF and 
                    name != "Unknown" and 
                    (current_time - last_analysis_time.get(name, 0)) > ANALYSIS_COOLDOWN and 
                    not is_recording): 
                    
                    last_analysis_time[name] = current_time
                    print(f"\nTriggering single-frame analysis for {name}...") 
                    analysis_thread = threading.Thread(target=analyze_frame_with_gemma, args=(frame.copy(), name), daemon=True)
                    analysis_thread.start()
                
                current_face_data.append((face_location, name, confidence))

            with data_lock:
                analysis_snapshot = analysis_results.copy()
            
            for (face_location, name, confidence) in current_face_data:
                if current_model == MODEL_OFF or is_recording:
                    analysis_text = "<i>(Paused)</i>"
                else:
                    analysis_text = analysis_snapshot.get(name, "")
                
                live_face_payload.append({
                    "name": name,
                    "confidence": int(confidence),
                    "analysis": analysis_text
                })
            
            with data_lock:
                server_data["live_faces"] = live_face_payload

        for (face_location, name, confidence) in current_face_data:
            top, right, bottom, left = face_location
            top = top - BOX_PADDING; right = right + BOX_PADDING
            bottom = bottom + BOX_PADDING; left = left - BOX_PADDING

            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label_text = f"{name} ({int(confidence)}%)"
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_top = max(top - text_height - baseline - 10, 0)
            label_bottom = max(top - 10, text_height + baseline)
            label_left = left; label_right = left + text_width + 10
            cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), color, cv2.FILLED)
            cv2.putText(frame, label_text, (left + 5, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        with data_lock:
            global output_frame
            output_frame = frame.copy()

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """(Unchanged) Streams the processed video frames."""
    def generate_frames():
        global output_frame, data_lock
        while True:
            with data_lock:
                if output_frame is None:
                    time.sleep(0.1); continue
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                if not flag: continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encoded_image) + b'\r\n')
            time.sleep(0.03)
            
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- ✨ --- UPDATED: /get_data Route --- ✨ ---
@app.route('/get_data')
def get_data():
    """Provides the live data to the frontend and adds no-cache headers."""
    global server_data, data_lock
    with data_lock:
        response_data = server_data.copy()
    
    response = jsonify(response_data)
    
    # ✨ --- NEW: Add no-cache headers to prevent browser caching --- ✨
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

@app.route('/toggle_action', methods=['POST'])
def toggle_action():
    """(Unchanged) Handles the button click to start/stop the *live action thread*."""
    global server_data, data_lock, action_thread, stop_action_thread, action_frames, last_action_frame_gray
    
    print("[Debug] /toggle_action route called")
    
    with data_lock:
        server_data["is_recording"] = not server_data["is_recording"]
        current_model = server_data['model'] 
        
        if server_data["is_recording"]:
            if current_model == MODEL_OFF:
                print("[Action Analysis] Blocked: Model is set to OFF.")
                server_data["action_result"] = "Model is Off. Please select a model to start."
                server_data["is_recording"] = False
            else:
                print("[Action Analysis] STARTING live comprehension thread.")
                stop_action_thread = False
                action_frames = [] 
                last_action_frame_gray = None 
                server_data["action_result"] = "Live analysis started...\n"
                server_data["keyframe_count"] = 0
                action_thread = threading.Thread(target=action_comprehension_thread, daemon=True)
                action_thread.start()
        else:
            print("[Action Analysis] STOPPING live comprehension thread.")
            if action_thread is not None:
                stop_action_thread = True 
                action_thread = None
            if server_data["action_result"] and "stopped" not in server_data["action_result"]:
                server_data["action_result"] += "\n...Live analysis stopped."
    
    return jsonify({"status": "ok", "is_recording": server_data["is_recording"]})

@app.route('/set_model', methods=['POST'])
def set_model():
    """(Unchanged) Handles the model switch from the frontend."""
    global server_data, data_lock
    
    data = request.json
    model_name = data.get('model', MODEL_GEMMA) 
    
    if model_name not in [MODEL_GEMMA, MODEL_LLAVA, MODEL_MOONDREAM, MODEL_OFF]:
        model_name = MODEL_GEMMA
        
    with data_lock:
        server_data['model'] = model_name
        if model_name == MODEL_OFF and server_data["is_recording"]:
            server_data["is_recording"] = False
            global stop_action_thread, action_thread
            stop_action_thread = True
            action_thread = None
            server_data["action_result"] += "\n...Model turned off. Analysis stopped."
    
    print(f"[Server] Model switched to: {model_name}")
    return jsonify({"status": "ok", "model": model_name})

@app.route('/set_detector', methods=['POST'])
def set_detector():
    """(Unchanged) Handles the detector switch from the frontend."""
    global server_data, data_lock
    
    data = request.json
    detector_name = data.get('detector', 'accurate') 
    
    if detector_name not in ['accurate', 'fast']:
        detector_name = 'accurate'
        
    with data_lock:
        server_data['detection_mode'] = detector_name
    
    print(f"[Server] Detector switched to: {detector_name}")
    return jsonify({"status": "ok", "detector": detector_name})

@app.route('/set_tuning', methods=['POST'])
def set_tuning():
    """(Unchanged) Handles the tuning sliders from the frontend."""
    global server_data, data_lock
    
    data = request.json
    
    with data_lock:
        try:
            server_data['nth_frame'] = int(data.get('nth_frame', server_data['nth_frame']))
            server_data['hog_scale_factor'] = float(data.get('hog_scale_factor', server_data['hog_scale_factor']))
        except Exception as e:
            print(f"Error updating tuning: {e}")
    
    return jsonify({"status": "ok"})

# --- Main entry point (Unchanged) ---
if __name__ == "__main__":
    load_known_faces(KNOWN_FACES_DIR)
    
    print("Starting webcam processing thread...")
    t = threading.Thread(target=video_processing_thread, daemon=True)
    t.start()
    
    print("Starting Flask server... open http://127.0.0.1:8080 in your browser.")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)