import cv2
import face_recognition # Still used for HOG detector and Dlib fallback
import os
import numpy as np
import ollama
import base64
import threading
import time
from retinaface import RetinaFace
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

# --- New SOTA Imports (with better error handling) ---
try:
    print("--- Attempting to import SOTA libraries... ---")
    from deepface import DeepFace
    print("1/2: DeepFace imported successfully.")
    # This is the correct module path for modern DeepFace versions
    from deepface.modules import verification as dst 
    print("2/2: deepface.modules.verification imported successfully.")
    SOTA_AVAILABLE = True
    print("--- SOTA libraries loaded successfully. ---")
except ImportError as e:
    print("\n!!! SOTA IMPORT FAILED (ImportError) !!!")
    print(f"!!! THE REAL ERROR IS: {e}")
    print("!!! This is likely a dependency issue *inside* DeepFace.")
    print("!!! Falling back to Dlib (face_recognition).\n")
    SOTA_AVAILABLE = False
except Exception as e:
     print(f"\n!!! SOTA IMPORT FAILED (Unexpected Error: {type(e).__name__}) !!!")
     print(f"!!! THE REAL ERROR IS: {e}")
     print("!!! This could be a version conflict or a backend issue (e.g., TensorFlow).")
     print("!!! Falling back to Dlib (face_recognition).\n")
     SOTA_AVAILABLE = False
# ---------------------------


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

# --- ✨ ASCII Progress Bar Helper ---
def print_progress_bar(progress, total_width=40, prefix="Progress:", suffix="Complete"):
    """Displays or updates a terminal progress bar."""
    filled_length = int(total_width * progress)
    bar = '█' * filled_length + '-' * (total_width - filled_length)
    # Use \r to return to the start of the line, end='' to not add a newline
    # flush=True is important to make it update immediately
    print(f'\r{prefix} |{bar}| {int(progress * 100)}% {suffix}', end='', flush=True)
    if progress >= 1:
        print() # Print a newline when done

# --- SOTA Face Recognition Config ---
if SOTA_AVAILABLE:
    SOTA_MODEL = "ArcFace"
    SOTA_METRIC = "cosine"
    try:
        print("1/3: Loading SOTA Face Model...")
        print_progress_bar(0, prefix=f"Loading {SOTA_MODEL}:", suffix="Starting...")
        
        DeepFace.build_model(SOTA_MODEL) # The actual load
        
        # Print "Done"
        print_progress_bar(1, prefix=f"Loading {SOTA_MODEL}:", suffix="Complete! ")
        
        MAX_RECOGNITION_DISTANCE = dst.find_threshold(SOTA_MODEL, SOTA_METRIC) 
        print(f"Max recognition distance ({SOTA_METRIC}) set to: {MAX_RECOGNITION_DISTANCE}")
    except Exception as e:
        print(f"!!! FATAL: Could not load {SOTA_MODEL}. Error: {e}")
        print("!!! Reverting to Dlib (face_recognition) defaults.")
        SOTA_MODEL = "Dlib"
        SOTA_METRIC = "euclidean"
        MAX_RECOGNITION_DISTANCE = 0.6
else:
    SOTA_MODEL = "Dlib" # Fallback
    SOTA_METRIC = "euclidean"
    MAX_RECOGNITION_DISTANCE = 0.6 # Original Dlib default

DEFAULT_RECOGNITION_THRESHOLD = MAX_RECOGNITION_DISTANCE # Default slider to the max
DEFAULT_RETINAFACE_CONF = 0.9 
ACTION_MOTION_THRESHOLD = 30
FACE_RECOGNITION_NTH_FRAME = 5 
HOG_SCALE_FACTOR = 0.5 

# --- Drawing Colors ---
COLOR_BODY_KNOWN = (255, 100, 100) # Light Blue
COLOR_BODY_UNKNOWN = (100, 100, 255) # Light Red
COLOR_FACE_BOX = (0, 255, 255) # Yellow
COLOR_TEXT_BACKGROUND = (0, 0, 0) # Black
COLOR_TEXT_FOREGROUND = (255, 255, 255) # White
# ---------------------

# --- YOLO Model Definitions ---
YOLO_MODELS = {
    "n": "yolo11n.pt", # Nano (Fastest)
    "s": "yolo11s.pt", # Small (Balanced)
    "m": "yolo11m.pt", # Medium (Accurate)
    "x": "yolo11x.pt"  # ✨ --- NEW: Ultra ---
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
    "yolo_imgsz": 640,
    "retinaface_conf": DEFAULT_RETINAFACE_CONF,
    "face_detector_mode": "accurate",
    "recognition_threshold": DEFAULT_RECOGNITION_THRESHOLD 
}

# --- Global ML State (Loaded once) ---
known_face_encodings = []
known_face_names = []
analysis_results = {}
action_thread = None
stop_action_thread = False
action_frames = []
last_action_frame_gray = None

print(f"2/3: Loading initial YOLO model...")
yolo_model_name = YOLO_MODELS[server_data['yolo_model_key']]
print_progress_bar(0, prefix=f"Loading {yolo_model_name}:", suffix="Starting...")

yolo_model = YOLO(yolo_model_name) # The actual load

print_progress_bar(1, prefix=f"Loading {yolo_model_name}:", suffix="Complete! ")

person_registry = {} # { track_id: {"name": "Gabriel", "confidence": 90, "face_location": (t,r,b,l)} }

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Functions (Frame Conversion, AI threads, etc.) ---

def frame_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    if not success: return None
    return base64.b64encode(buffer).decode('utf-8')

def action_comprehension_thread():
    global data_lock, output_frame, server_data, stop_action_thread, last_action_frame_gray
    print("[Action Thread] Started.")
    chat_messages = []
    last_frame_gray = None
    keyframe_count = 0
    
    # Use the global last_action_frame_gray to sync with main thread
    with data_lock:
        last_frame_gray = last_action_frame_gray 

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
            frame_delta = cv2.absdiff(last_frame_gray, gray)
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

# --- NEW HELPER for load_known_faces ---
def load_with_dlib(known_faces_dir, all_people_dirs):
    """Helper to load faces using Dlib and show progress."""
    global known_face_encodings, known_face_names
    
    total_people = len(all_people_dirs)
    if total_people == 0: return

    print_progress_bar(0, prefix=f"Loading Dlib Faces:", suffix=f"(0/{total_people})")
    
    # Clear any partial SOTA loads
    known_face_encodings.clear()
    known_face_names.clear()
    
    for i, (person_name, person_dir) in enumerate(all_people_dirs):
        image_count = 0
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(person_dir, filename)
                try:
                    face_image = face_recognition.load_image_file(image_path)
                    face_encodings_dlib = face_recognition.face_encodings(face_image)
                    for encoding in face_encodings_dlib:
                        known_face_encodings.append(encoding)
                        known_face_names.append(person_name)
                    image_count += 1
                except Exception as e:
                    # Supress error during progress bar to avoid visual glitches
                    # print(f"\n - ERROR loading {image_path} with Dlib: {e}")
                    pass
        
        # Update progress bar
        progress = (i + 1) / total_people
        print_progress_bar(progress, prefix=f"Loading Dlib Faces:", suffix=f"({i+1}/{total_people})")

# --- UPDATED: SOTA-Aware Face Loader ---
def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names, SOTA_MODEL, SOTA_METRIC, MAX_RECOGNITION_DISTANCE, DEFAULT_RECOGNITION_THRESHOLD
    
    print(f"3/3: Loading known faces from {known_faces_dir}...")
    
    # --- Pre-scan to get total count for progress bar ---
    all_people_dirs = []
    if os.path.exists(known_faces_dir):
        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if os.path.isdir(person_dir) and not person_name.startswith('.'):
                all_people_dirs.append((person_name, person_dir))
    
    if not all_people_dirs:
        print("No known face folders found.")
        return
    
    total_people = len(all_people_dirs)
    
    # --- SOTA path ---
    if SOTA_MODEL != "Dlib":
        print_progress_bar(0, prefix=f"Loading SOTA Faces:", suffix=f"(0/{total_people})")
        
        for i, (person_name, person_dir) in enumerate(all_people_dirs):
            image_count = 0
            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        # Use DeepFace.represent to get the embedding
                        representation = DeepFace.represent(
                            img_path=image_path, 
                            model_name=SOTA_MODEL,
                            enforce_detection=True,
                            detector_backend='retinaface' # Use a good detector
                        )
                        
                        if representation and len(representation) > 0:
                            embedding = representation[0]["embedding"]
                            known_face_encodings.append(embedding)
                            known_face_names.append(person_name)
                            image_count += 1
                        else:
                            pass # Supress warning for cleaner progress bar

                    except Exception as e:
                        pass # Supress error for cleaner progress bar
            
            # Update progress bar
            progress = (i + 1) / total_people
            print_progress_bar(progress, prefix=f"Loading SOTA Faces:", suffix=f"({i+1}/{total_people})")

        # If SOTA loading failed entirely, fall back to Dlib
        if not known_face_encodings:
            print("\n!!! WARNING: SOTA load failed, falling back to Dlib...")
            SOTA_MODEL = "Dlib"
            SOTA_METRIC = "euclidean"
            MAX_RECOGNITION_DISTANCE = 0.6
            DEFAULT_RECOGNITION_THRESHOLD = 0.6
            with data_lock:
                 server_data["recognition_threshold"] = DEFAULT_RECOGNITION_THRESHOLD
            load_with_dlib(known_faces_dir, all_people_dirs)

    # Dlib loading (runs if SOTA_MODEL was "Dlib" initially)
    else:
        load_with_dlib(known_faces_dir, all_people_dirs)


def get_containing_body_box(face_box, body_boxes):
    ft, fr, fb, fl = face_box
    face_center_x = (fl + fr) / 2
    face_center_y = (ft + fb) / 2
    for track_id, (bt, br, bb, bl) in body_boxes.items():
        if bl < face_center_x < br and bt < face_center_y < bb:
            return track_id
    return None

# --- UPDATED: SOTA-Aware Video Processing Thread ---
def video_processing_thread():
    """
    Main background thread.
    Uses YOLO for body tracking + (RetinaFace OR HOG) for face recognition.
    """
    global data_lock, output_frame, server_data
    global analysis_results, yolo_model, person_registry
    global last_action_frame_gray, action_frames
    
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
            current_yolo_conf = server_data["yolo_conf"]
            current_yolo_imgsz = server_data["yolo_imgsz"]
            current_retinaface_conf = server_data["retinaface_conf"]
            current_face_detector = server_data["face_detector_mode"]
            current_recognition_threshold = server_data["recognition_threshold"] 

        if is_recording:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if last_action_frame_gray is None:
                print(f"[Action Analysis] Set Keyframe 1 (Start)")
                last_action_frame_gray = gray
            else:
                frame_delta = cv2.absdiff(last_action_frame_gray, gray); thresh = cv2.threshold(frame_delta, ACTION_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                if thresh.sum() > 0:
                    last_action_frame_gray = gray; # This frame becomes the new baseline for motion
                    
        # --- YOLO-BASED TRACKING ---
        
        # 1. Detect & Track Bodies (Every Frame)
        yolo_results = yolo_model.track(
            frame, 
            persist=True, 
            classes=[0], # 0 = "person"
            conf=current_yolo_conf, 
            imgsz=current_yolo_imgsz, 
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
            
            face_locations = []
            
            # --- Clear old face locations before re-detection ---
            for track_id in person_registry:
                if "face_location" in person_registry[track_id]:
                    person_registry[track_id]["face_location"] = None

            # --- STEP A: DETECT FACE LOCATIONS ---
            if current_face_detector == "fast":
                # --- FAST (HOG) MODE ---
                scale = HOG_SCALE_FACTOR
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations_small = face_recognition.face_locations(rgb_small_frame, model='hog')
                draw_scale = 1 / scale
                face_locations = [(int(t*draw_scale), int(r*draw_scale), int(b*draw_scale), int(l*draw_scale)) for (t,r,b,l) in face_locations_small]
            
            else:
                # --- ACCURATE (RETINAFACE) MODE ---
                try:
                    faces = RetinaFace.detect_faces(frame, threshold=current_retinaface_conf) 
                except Exception as e: 
                    print(f"RetinaFace Error: {e}"); faces = {}
                
                if isinstance(faces, dict):
                    for face_key, face_data in faces.items():
                        x1, y1, x2, y2 = face_data['facial_area']
                        face_locations.append((int(y1), int(x2), int(y2), int(x1))) # t,r,b,l

            # --- STEP B: ENCODE & COMPARE FACES ---
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Needed for both paths

            for face_location in face_locations:
                name, confidence = "Unknown", 0
                current_face_encoding = None

                # --- Get SOTA Encoding ---
                if SOTA_MODEL != "Dlib":
                    try:
                        # Crop the face from the RGB frame: [t, b, l, r]
                        t, r, b, l = face_location
                        # Add padding to ensure DeepFace gets the whole face
                        t_pad = max(0, t - BOX_PADDING)
                        b_pad = min(frame.shape[0], b + BOX_PADDING)
                        l_pad = max(0, l - BOX_PADDING)
                        r_pad = min(frame.shape[1], r + BOX_PADDING)
                        face_crop = rgb_frame[t_pad:b_pad, l_pad:r_pad]
                        
                        if face_crop.size == 0: continue
                        
                        # Get SOTA embedding
                        representation = DeepFace.represent(
                            img_path=face_crop,
                            model_name=SOTA_MODEL,
                            enforce_detection=False, # We already detected
                            detector_backend='skip'
                        )
                        current_face_encoding = representation[0]["embedding"]
                    except Exception as e:
                        # print(f"DeepFace encode error: {e}") # Can be noisy
                        pass # Skip this face if encoding fails
                
                # --- Get Dlib Encoding (Fallback) ---
                else:
                    # Use the original (fast) dlib method
                    encodings = face_recognition.face_encodings(rgb_frame, [face_location])
                    if encodings:
                        current_face_encoding = encodings[0]
                
                # --- STEP C: COMPARE ENCODINGS ---
                if current_face_encoding is not None and len(known_face_encodings) > 0:
                    
                    # --- SOTA-AWARE DISTANCE CALCULATION ---
                    if SOTA_METRIC == "cosine":
                        # Use DeepFace's internal cosine function
                        face_distances = [dst.find_cosine_distance(known_encoding, current_face_encoding) for known_encoding in known_face_encodings]
                    else: # Default to Euclidean (for Dlib)
                        face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        min_distance = face_distances[best_match_index]
                        
                        # Compare against the *dynamic* threshold from the slider
                        if min_distance < current_recognition_threshold:
                            name = known_face_names[best_match_index]
                            # Use the SOTA-aware MAX distance for scaling confidence
                            confidence = max(0, min(100, (1.0 - (min_distance / MAX_RECOGNITION_DISTANCE)) * 100))
                    # --- END SOTA-AWARE DISTANCE ---

                # --- STEP D: ASSOCIATE & ANALYZE ---
                if name != "Unknown":
                    # 3. Association Logic
                    body_track_id = get_containing_body_box(face_location, body_boxes_with_ids)
                    if body_track_id is not None:
                        # 4. "Remember" the Name and Face Location
                        # print(f"[Registry] Associated {name} ({SOTA_MODEL}) with Person ID {body_track_id}")
                        person_registry[body_track_id] = {
                            "name": name, 
                            "confidence": confidence, 
                            "face_location": face_location 
                        }
                    
                    # 5. Trigger single-frame analysis
                    current_time = time.time()
                    if (current_model != MODEL_OFF and 
                        (current_time - last_analysis_time.get(name, 0)) > ANALYSIS_COOLDOWN and 
                        not is_recording): 
                        last_analysis_time[name] = current_time
                        print(f"\nTriggering single-frame analysis for {name}...") 
                        analysis_thread = threading.Thread(target=analyze_frame_with_gemma, args=(frame.copy(), name), daemon=True)
                        analysis_thread.start()

        # 6. Drawing & Data Sync (Every Frame)
        live_face_payload = []
        with data_lock:
            analysis_snapshot = analysis_results.copy()

        for track_id, (t, r, b, l) in body_boxes_with_ids.items():
            person_info = person_registry.get(track_id, {"name": "Person", "confidence": 0.0, "face_location": None})
            name = person_info["name"]
            confidence = person_info["confidence"]
            face_location = person_info["face_location"] 

            # --- Drawing Logic (Body) ---
            body_color = COLOR_BODY_KNOWN if name != "Person" else COLOR_BODY_UNKNOWN
            t_body, r_body, b_body, l_body = max(0, t-BOX_PADDING), min(frame.shape[1], r+BOX_PADDING), min(frame.shape[0], b+BOX_PADDING), max(0, l-BOX_PADDING)
            cv2.rectangle(frame, (l_body, t_body), (r_body, b_body), body_color, 2)
            
            label_text = f"{name}"
            if confidence > 0 and name != "Person": 
                label_text += f" ({int(confidence)}%)"
            elif name == "Person":
                label_text = "Unknown Person"
            
            # New Drawing Logic for Text Background and Text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 # Slightly larger font
            font_thickness = 2 # Thicker font
            text_padding = 5 # Padding around the text
            
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Position the text background rectangle
            # It should be above the body box, aligned to its left edge
            label_top = max(0, t_body - text_height - baseline - text_padding * 2)
            label_bottom = label_top + text_height + baseline + text_padding * 2
            label_left = l_body
            label_right = l_body + text_width + text_padding * 2

            # Draw the filled background rectangle
            cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), body_color, cv2.FILLED)
            # Draw the text
            cv2.putText(frame, label_text, (l_body + text_padding, label_bottom - text_padding - baseline), font, font_scale, COLOR_TEXT_FOREGROUND, font_thickness)

            # --- Drawing Logic (Face) --- 
            if face_location: 
                ft, fr, fb, fl = face_location
                cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

            # --- Payload Logic (Unchanged) ---
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
    
    # print("[Debug] /toggle_action route called")
    
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
                last_action_frame_gray = None # Reset motion detector
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
        
        if new_model_key != server_data['yolo_model_key']:
            model_path = YOLO_MODELS.get(new_model_key, "yolo11n.pt") 
            print(f"[Server] Loading new YOLO model: {model_path}...")
            try:
                yolo_model = YOLO(model_path)
                server_data['yolo_model_key'] = new_model_key
                print("[Server] New YOLO model loaded.")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                server_data['yolo_model_key'] = server_data['yolo_model_key']
        
        server_data['yolo_conf'] = float(data.get('yolo_conf', server_data['yolo_conf']))
        server_data['yolo_imgsz'] = int(data.get('yolo_imgsz', server_data['yolo_imgsz']))

    # print(f"[Server] YOLO settings updated: {server_data['yolo_model_key']}, {server_data['yolo_conf']}, {server_data['yolo_imgsz']}")
    return jsonify({"status": "ok"})

@app.route('/set_retinaface_tuning', methods=['POST'])
def set_retinaface_tuning():
    """Handles the RetinaFace confidence slider."""
    global server_data, data_lock
    data = request.json
    with data_lock:
        try:
            server_data['retinaface_conf'] = float(data.get('retinaface_conf', server_data['retinaface_conf']))
        except Exception as e:
            print(f"Error updating RetinaFace conf: {e}")
    return jsonify({"status": "ok"})

@app.route('/set_face_detector', methods=['POST'])
def set_face_detector():
    """Handles the face detector switch from the frontend."""
    global server_data, data_lock
    
    data = request.json
    detector_name = data.get('face_detector_mode', 'accurate') 
    
    if detector_name not in ['accurate', 'fast']:
        detector_name = 'accurate'
        
    with data_lock:
        server_data['face_detector_mode'] = detector_name
    
    print(f"[Server] Face Detector switched to: {detector_name}")
    return jsonify({"status": "ok", "face_detector_mode": detector_name})

@app.route('/set_recognition_tuning', methods=['POST'])
def set_recognition_tuning():
    """Handles the face recognition threshold slider."""
    global server_data, data_lock
    
    data = request.json
    
    with data_lock:
        try:
            server_data['recognition_threshold'] = float(data.get('recognition_threshold', server_data['recognition_threshold']))
        except Exception as e:
            print(f"Error updating recognition threshold: {e}")
    
    return jsonify({"status": "ok"})

# --- Main entry point ---
if __name__ == "__main__":
    load_known_faces(KNOWN_FACES_DIR)
    
    print("All models loaded. Starting webcam processing thread...")
    t = threading.Thread(target=video_processing_thread, daemon=True)
    t.start()
    
    print("Starting Flask server... open http://127.0.0.1:8080 in your browser.")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)