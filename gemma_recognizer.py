import cv2
import face_recognition
import os
import numpy as np
import ollama
import base64
import threading
import time

# --- Configuration ---
MODEL = 'gemma3:4b'          # The Ollama model to use for analysis
WEBCAM_INDEX = 0             # 0 is usually the default (built-in) webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces" # Folder containing subfolders (e.g., 'Gabriel', 'Mark')
ANALYSIS_COOLDOWN = 10       # Seconds before re-analyzing the same person

# --- Tuning Variables ---
PROCESS_SCALE_FACTOR = 0.5
PROCESS_EVERY_NTH_FRAME = 2

# --- Drawing Variables ---
BOX_PADDING = 15
CONFIDENCE_THRESHOLD = 0.6

# --- ✨ New Action Analysis Variables ✨ ---
# Capture every 5th frame for the action sequence
ACTION_FRAME_CAPTURE_RATE = 5
# Define the button's position and size (x, y, width, height)
BUTTON_RECT = (10, FRAME_HEIGHT - 50, 200, 40)
# The window name MUST be a constant to register the mouse callback
WINDOW_NAME = 'Gemma 3 Facial Recognition'
# ---------------------

# --- Global Dictionaries & State ---
analysis_results = {}        # For single-frame analysis
last_analysis_time = {}      # For single-frame analysis
is_analyzing_action = False  # Toggle for the new button
action_frames = []           # List to store base64 frames for action analysis
action_analysis_result = ""  # String to hold the final action report
# ---------------------------

def frame_to_base64(frame):
    """Converts an OpenCV frame (BGR image) to a base64 string."""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

# --- ✨ New Function for Action Analysis ✨ ---
def analyze_action_sequence(frames_list):
    """
    Function to run in a separate thread.
    Sends a *sequence* of frames to Gemma 3 for complex action analysis.
    """
    global action_analysis_result
    
    if not frames_list:
        action_analysis_result = "Error: No frames captured for action."
        return

    action_analysis_result = f"Analyzing {len(frames_list)} frames..."
    print(f"\n[Action Analysis] Sending {len(frames_list)} frames to Gemma 3...")

    prompt = (
        "This is a sequence of video frames captured in order. "
        "Analyze the entire sequence and describe the complex action or event that occurred. "
        "What is the person or people doing from start to finish?"
    )

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': frames_list  # Pass the entire list of base64 images
                }
            ],
            stream=False
        )
        
        analysis_text = response['message']['content']
        
        print(f"\n--- 🔳 Action Analysis Result 🔳 ---\n{analysis_text}\n---------------------------------")
        
        action_analysis_result = analysis_text
        
    except Exception as e:
        print(f"Error calling Ollama for action analysis: {e}")
        action_analysis_result = "Error: Ollama connection failed."

# --- ✨ New Mouse Callback Function for the Button ✨ ---
def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to toggle the action analysis."""
    global is_analyzing_action, action_frames, action_analysis_result

    # Check if the click is a left-button-down event
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click (x, y) is inside our button's rectangle
        bx, by, bw, bh = BUTTON_RECT
        if bx <= x <= bx + bw and by <= y <= by + bh:
            
            # Toggle the action analysis state
            is_analyzing_action = not is_analyzing_action
            
            if is_analyzing_action:
                # --- STARTING ---
                print("[Action Analysis] Started recording frames.")
                action_frames = []  # Clear previous frames
                action_analysis_result = "" # Clear previous result
            else:
                # --- STOPPING ---
                print(f"[Action Analysis] Stopped recording. Captured {len(action_frames)} frames.")
                if len(action_frames) > 0:
                    # Start the analysis in a new thread
                    analysis_thread = threading.Thread(
                        target=analyze_action_sequence,
                        args=(action_frames.copy(),), # Pass a copy
                        daemon=True
                    )
                    analysis_thread.start()
                else:
                    action_analysis_result = "No frames captured."

# --- This function is unchanged ---
def analyze_frame_with_gemma(frame, name):
    """(Unchanged) Analyzes a *single* frame for face recognition context."""
    global analysis_results
    
    b64_image = frame_to_base64(frame)
    if b64_image is None:
        analysis_results[name] = "Error: Failed to encode frame."
        return

    prompt = f"This person has been identified as {name}. Briefly describe what they are doing."

    try:
        response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt, 'images': [b64_image]}], stream=False)
        analysis_text = response['message']['content']
        print(f"\n--- Analysis for {name} ---\n{analysis_text}\n--------------------")
        analysis_results[name] = analysis_text
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        analysis_results[name] = "Error connecting to Ollama."

# --- This function is unchanged ---
def load_known_faces(known_faces_dir):
    """(Unchanged) Loads face encodings from subfolders."""
    known_face_encodings = []
    known_face_names = []
    
    print(f"Loading known faces from {known_faces_dir}...")
    if not os.path.exists(known_faces_dir):
        print(f"Error: Directory not found: {known_faces_dir}")
        print("Please create it and add subfolders for each person.")
        return [], []
        
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir) or person_name.startswith('.'):
            continue
            
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

    return known_face_encodings, known_face_names

# --- ✨ Main Function is Updated ---
def main():
    # 1. Load known faces (unchanged)
    known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)
    
    if not known_face_encodings:
        print("No known faces loaded. The program will only identify 'Unknown' persons.")

    # 2. Open webcam (unchanged)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # --- ✨ New Window and Callback Setup ✨ ---
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    print("Webcam stream started. Press 'q' in the OpenCV window to quit.")
    print("Click the 'START' button in the window to record an action.")
    
    draw_scale_factor = 1 / PROCESS_SCALE_FACTOR
    frame_count = 0
    current_face_data = [] # List of (face_location, name, confidence)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break
            
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # --- ✨ New: Capture Action Frames if Recording ✨ ---
        if is_analyzing_action and (frame_count % ACTION_FRAME_CAPTURE_RATE == 0):
            b64_frame = frame_to_base64(frame)
            if b64_frame:
                action_frames.append(b64_frame)

        # --- Face Recognition Logic (unchanged) ---
        if frame_count % PROCESS_EVERY_NTH_FRAME == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE_FACTOR, fy=PROCESS_SCALE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current_face_data = [] 

            for face_location, face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                confidence = 0
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]

                    if min_distance < CONFIDENCE_THRESHOLD:
                        name = known_face_names[best_match_index]
                        confidence = max(0, min(100, (1.0 - min_distance) * 100))
                    else:
                        confidence = max(0, min(100, (1.0 - min_distance) * 100))
                
                current_face_data.append((face_location, name, confidence))

            # --- Hybrid Logic (Trigger Single-Frame Analysis - unchanged) ---
            current_time = time.time()
            for face_location, name, confidence in current_face_data:
                if name == "Unknown": continue 
                
                last_seen = last_analysis_time.get(name, 0)
                if (current_time - last_seen) > ANALYSIS_COOLDOWN:
                    last_analysis_time[name] = current_time
                    analysis_results[name] = "Analyzing..."
                    
                    print(f"\nTriggering Gemma 3 analysis for {name}...")
                    analysis_thread = threading.Thread(target=analyze_frame_with_gemma, args=(frame.copy(), name), daemon=True)
                    analysis_thread.start()

        # --- Face Recognition Drawing Logic (unchanged) ---
        for face_location, name, confidence in current_face_data:
            top, right, bottom, left = face_location
            top = int(top * draw_scale_factor) - BOX_PADDING
            right = int(right * draw_scale_factor) + BOX_PADDING
            bottom = int(bottom * draw_scale_factor) + BOX_PADDING
            left = int(left * draw_scale_factor) - BOX_PADDING
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label_text = f"{name} ({int(confidence)}%)"
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_top = max(top - text_height - baseline - 10, 0)
            label_bottom = max(top - 10, text_height + baseline)
            label_left = left
            label_right = left + text_width + 10
            cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), color, cv2.FILLED)
            cv2.putText(frame, label_text, (left + 5, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if name in analysis_results:
                analysis_text = analysis_results[name]
                analysis_y_start = max(label_bottom + 10, 50)
                y_offset = 0 
                for i, line in enumerate(analysis_text.split('\n')):
                    font_scale = 0.6; line_height = int(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][1] * 1.5)
                    y = analysis_y_start + y_offset
                    if y > FRAME_HEIGHT - 20: break
                    cv2.putText(frame, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
                    cv2.putText(frame, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                    y_offset += line_height

        # --- ✨ New Drawing Logic for Button and Action Result ✨ ---
        
        # 1. Draw the button
        bx, by, bw, bh = BUTTON_RECT
        if is_analyzing_action:
            # Recording: Red button, "STOP" text
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "STOP Recording", (bx + 10, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Show a recording indicator
            rec_text = f"REC {len(action_frames)}"
            cv2.putText(frame, rec_text, (bx + bw + 10, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Not recording: Green button, "START" text
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 180, 0), cv2.FILLED)
            cv2.putText(frame, "Analyse Action", (bx + 10, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 2. Draw the action analysis result (in the top left)
        if action_analysis_result:
            y_offset = 30
            for i, line in enumerate(action_analysis_result.split('\n')):
                y = y_offset
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) # Black outline
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow text
                y_offset += 20 # Move down for the next line

        # --- Display the final frame ---
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == ord('q'):
            print("\nQuitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream ended.")

if __name__ == "__main__":
    main()