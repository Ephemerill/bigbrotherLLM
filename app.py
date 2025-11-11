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
from ultralytics import YOLO # Import YOLO

# --- Configuration ---
MODEL_GEMMA = 'gemma3:4b'
MODEL_LLAVA = 'llava'
MODEL_MOONDREAM = 'moondream'
MODEL_OFF = 'off'
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
ANALYSIS_COOLDOWN = 10
BOX_PADDING = 10
CONFIDENCE_THRESHOLD = 0.6
DETECTOR_CONFIDENCE = 0.9 # For RetinaFace
ACTION_MOTION_THRESHOLD = 30
# Run slow face-recog only every 5 frames
FACE_RECOGNITION_NTH_FRAME = 5 
# ---------------------

# --- YOLO Model Definitions ---
YOLO_MODELS = {
    "n": "yolov8n.pt", # Nano (Fastest)
    "s": "yolov8s.pt", # Small (Balanced)
    "m": "yolov8m.pt"  # Medium (Accurate)
}

# --- Global Server State ---
data_lock = threading.Lock()
output_frame = None
server_data = {
    "is_recording": False,
    "keyframe_count": 0,
    "action_result": "",
    "live_faces": [],
    "model": MODEL_GEMMA,
    "yolo_model_key": "n",
    "yolo_conf": 0.4,
    "yolo_imgsz": 640
}

# --- Global ML State (Loaded once) ---
known_face_encodings = []
known_face_names = []
analysis_results = {}
action_thread = None
stop_action_thread = False

print(f"Loading initial YOLO model: {YOLO_MODELS[server_data['yolo_model_key']]}...")
yolo_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']])
print("YOLO model loaded.")

# This "remembers" who is who. Format: { track_id: "Gabriel" }
person_registry = {}

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Functions (Frame Conversion, AI threads, etc.) are UNCHANGED ---
# ... (frame_to_base64 is unchanged) ...
# ... (action_comprehension_thread is unchanged) ...
# ... (analyze_frame_with_gemma is unchanged) ...
# ... (load_known_faces is unchanged) ...

def frame_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    if not success: return None
    return base64.b64encode(buffer).decode('utf-8')

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
            current_model = server_data['model']
        if current_frame is None:
            time.sleep(0.1); continue
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
        is_keyframe = False
        if last_frame_gray is None:
            is_keyframe = True
        else:
            frame_delta = cv2.absdiff(last_action_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, ACTION_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            if thresh.sum() > 0: is_keyframe = True
        if is_keyframe:
            print(f"[Action Thread] Motion detected! Processing Keyframe {keyframe_count + 1}.")
            last_frame_gray = gray; keyframe_count += 1
            with data_lock: server_data["keyframe_count"] = keyframe_count
            b64_frame = frame_to_base64(current_frame)
            if not b64_frame: continue
            prompt = "";
            if current_model == MODEL_MOONDREAM:
                prompt = "What action is happening in this image?"
            else:
                if not chat_messages: prompt = "This is the first keyframe. Briefly describe what is happening."
                else: prompt = "This is the next keyframe. Briefly describe the new action."
            chat_messages.append({"role": "user", "content": prompt, "images": [b64_frame]})
            try:
                print(f"[Action Thread] Sending to {current_model}...")
                response = ollama.chat(model=current_model, messages=chat_messages, stream=False)
                response_message = response['message']; response_content = response_message['content']
                chat_messages.append(response_message)
                with data_lock: server_data["action_result"] += f"- {response_content}\n"
            except Exception as e:
                print(f"[Action Thread] Error calling Ollama: {e}")
                with data_lock: server_data["action_result"] = "Error connecting to Ollama."
                time.sleep(2)
        time.sleep(0.5) 
    print("[Action Thread] Stopped.")

def analyze_frame_with_gemma(frame, name):
    global analysis_results, data_lock
    b64_image = frame_to_base64(frame)
    if b64_image is None:
        with data_lock: analysis_results[name] = "Error: Failed to encode frame."
        return
    with data_lock: current_model = server_data['model']
    if current_model == MODEL_OFF:
        print("[Single Frame] Analysis skipped, model is OFF."); return
    prompt = ""
    if current_model == MODEL_MOONDREAM:
        prompt = "What is the person in this image doing?"
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

# --- Helper: Find which body a face belongs to (Unchanged) ---
def get_containing_body_box(face_box, body_boxes):
    """
    Finds the body box that a face box is inside of.
    face_box: (t, r, b, l)
    body_boxes: dict of {track_id: (t, r, b, l)}
    """
    ft, fr, fb, fl = face_box
    face_center_x = (fl + fr) / 2
    face_center_y = (ft + fb) / 2

    for track_id, (bt, br, bb, bl) in body_boxes.items():
        if bl < face_center_x < br and bt < face_center_y < bb:
            return track_id
    return None

# --- ✨ --- UPDATED: Video Processing Thread --- ✨ ---
def video_processing_thread():
    """
    Main background thread.
    Uses YOLO for body tracking + RetinaFace for face recognition.
    """
    global data_lock, output_frame, server_data
    global analysis_results, yolo_model, person_registry # ✨ --- ADDED person_registry --- ✨
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    frame_count = 0
    last_analysis_time = {} 
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Retrying...")
            time.sleep(1.0); continue
            
        frame = cv2.flip(frame, 1)
        frame_count += 1

        with data_lock:
            is_recording = server_data["is_recording"]
            current_model = server_data["model"]
            current_conf = server_data["yolo_conf"]
            current_imgsz = server_data["yolo_imgsz"]

        if is_recording:
            # ... (Keyframe logic is unchanged) ...
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
            global last_action_frame_gray, action_frames
            if last_action_frame_gray is None:
                last_action_frame_gray = gray; b64_frame = frame_to_base64(frame)
                if b64_frame: action_frames.append(b64_frame); print(f"[Action Analysis] Saved Keyframe 1 (Start)")
            else:
                frame_delta = cv2.absdiff(last_action_frame_gray, gray); thresh = cv2.threshold(frame_delta, ACTION_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                if thresh.sum() > 0:
                    last_action_frame_gray = gray; b64_frame = frame_to_base64(frame)
                    if b64_frame: action_frames.append(b64_frame); print(f"[Action Analysis] Saved Keyframe {len(action_frames)} (Motion Detected)")
            with data_lock: server_data["keyframe_count"] = len(action_frames)


        # --- YOLO-BASED TRACKING ---
        
        # 1. Detect & Track Bodies (Every Frame)
        yolo_results = yolo_model.track(
            frame, 
            persist=True, 
            classes=[0], # 0 = "person"
            conf=current_conf, 
            imgsz=current_imgsz, 
            verbose=False
        )
        
        body_boxes_with_ids = {} # {track_id: (t,r,b,l)}
        
        if yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, track_ids):
                l, t, r, b = box
                body_boxes_with_ids[track_id] = (t, r, b, l)

        # 2. Detect & Recognize Faces (Every Nth Frame)
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0:
            try:
                faces = RetinaFace.detect_faces(frame, threshold=DETECTOR_CONFIDENCE)
            except Exception as e: faces = {}
            
            face_locations = []
            if isinstance(faces, dict):
                for face_key, face_data in faces.items():
                    x1, y1, x2, y2 = face_data['facial_area']
                    face_locations.append((int(y1), int(x2), int(y2), int(x1))) # t,r,b,l
            
            if face_locations:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    
                    if name != "Unknown":
                        # 3. Association Logic
                        body_track_id = get_containing_body_box(face_location, body_boxes_with_ids)
                        if body_track_id is not None:
                            # 4. "Remember" the Name
                            print(f"[Registry] Associated {name} with Person ID {body_track_id}")
                            person_registry[body_track_id] = (name, confidence)
                        
                        # Trigger single-frame analysis
                        current_time = time.time()
                        if (current_model != MODEL_OFF and 
                            (current_time - last_analysis_time.get(name, 0)) > ANALYSIS_COOLDOWN and 
                            not is_recording): 
                            last_analysis_time[name] = current_time
                            print(f"\nTriggering single-frame analysis for {name}...") 
                            analysis_thread = threading.Thread(target=analyze_frame_with_gemma, args=(frame.copy(), name), daemon=True)
                            analysis_thread.start()

        # 5. Drawing & Data Sync (Every Frame)
        live_face_payload = []
        with data_lock:
            analysis_snapshot = analysis_results.copy()

        # Loop over the *tracked bodies* from YOLO
        for track_id, (t, r, b, l) in body_boxes_with_ids.items():
            # This is the line that was failing
            name, confidence = person_registry.get(track_id, ("Person", 0.0))

            # --- Drawing Logic (Body) ---
            color = (0, 255, 0) if name != "Person" else (255, 100, 100) # Green if known, blue if unknown
            t, r, b, l = t-BOX_PADDING, r+BOX_PADDING, b+BOX_PADDING, l-BOX_PADDING
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            label_text = f"{name}"
            if confidence > 0:
                label_text += f" ({int(confidence)}%)"
            
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_top = max(t - text_height - baseline - 10, 0)
            label_bottom = max(t - 10, text_height + baseline)
            label_left = l; label_right = l + text_width + 10
            cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), color, cv2.FILLED)
            cv2.putText(frame, label_text, (l + 5, t - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2) # Black text
            
            # --- Payload Logic ---
            if current_model == MODEL_OFF or is_recording:
                analysis_text = "<i>(Paused)</i>"
            else:
                analysis_text = analysis_snapshot.get(name, "")
            
            live_face_payload.append({
                "name": name,
                "confidence": int(confidence),
                "analysis": analysis_text
            })
        
        # --- FINAL STATE UPDATE (runs EVERY frame) ---
        with data_lock:
            global output_frame
            output_frame = frame.copy()
            server_data["live_faces"] = live_face_payload

# --- FLASK ROUTES (Unchanged) ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Streams the processed video frames."""
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

@app.route('/get_data')
def get_data():
    """Provides the live data to the frontend and adds no-cache headers."""
    global server_data, data_lock
    with data_lock:
        response_data = server_data.copy()
    
    response = jsonify(response_data)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/toggle_action', methods=['POST'])
def toggle_action():
    """Handles the button click to start/stop the *live action thread*."""
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
    """Handles the model switch from the frontend."""
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


@app.route('/set_yolo_settings', methods=['POST'])
def set_yolo_settings():
    """Handles the YOLO tuning from the frontend."""
    global server_data, data_lock, yolo_model
    
    data = request.json
    
    with data_lock:
        new_model_key = data.get('yolo_model_key', server_data['yolo_model_key'])
        
        # --- Check if we need to load a new model ---
        if new_model_key != server_data['yolo_model_key']:
            model_path = YOLO_MODELS.get(new_model_key, "yolov8n.pt")
            print(f"[Server] Loading new YOLO model: {model_path}...")
            try:
                yolo_model = YOLO(model_path)
                server_data['yolo_model_key'] = new_model_key
                print("[Server] New YOLO model loaded.")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                # Revert to old key if loading fails
                server_data['yolo_model_key'] = server_data['yolo_model_key']
        
        # Update conf and imgsz
        server_data['yolo_conf'] = float(data.get('yolo_conf', server_data['yolo_conf']))
        server_data['yolo_imgsz'] = int(data.get('yolo_imgsz', server_data['yolo_imgsz']))

    print(f"[Server] YOLO settings updated: {server_data['yolo_model_key']}, {server_data['yolo_conf']}, {server_data['yolo_imgsz']}")
    return jsonify({"status": "ok"})

# --- Main entry point ---
if __name__ == "__main__":
    load_known_faces(KNOWN_FACES_DIR)
    
    print("Starting webcam processing thread...")
    t = threading.Thread(target=video_processing_thread, daemon=True)
    t.start()
    
    print("Starting Flask server... open http://127.0.0.1:8080 in your browser.")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)