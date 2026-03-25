# Classroom Engagement Heatmap & Group Analysis
## Overview
This project analyzes classroom video data to estimate student engagement using computer vision techniques. It processes video frames, detects faces, estimates gaze and eye openness, computes engagement scores, and generates both JSON output and an HTML report with visualizations.

### estimate_gaze(face_crop)
This function estimates the gaze direction of a detected face.
Input
•	face_crop: Cropped image of a face
Process
•	Converts the face image to grayscale
•	Extracts the upper-middle region where eyes are likely located
•	Splits the eye region into left and right halves
•	Compares brightness between both halves
Logic
•	If brightness difference is small - forward
•	If left side brighter - gaze left
•	If right side brighter - gaze right
•	If no valid region - unknown
Output
•	Returns one of: forward, left, right, unknown

### estimate_eye_openness(face_crop)
This function estimates how open the eyes are.
Input
•	face_crop: Cropped face image
Process
•	Converts image to grayscale
•	Extracts eye region
•	Computes variance of pixel intensity
Logic
•	Higher variance - more texture - eyes likely open
•	Lower variance - flat region - possibly closed eyes
•	Value normalized to range 0-100
Output
•	Float value representing eye openness score

### score_face(face_crop)
This function computes engagement score for a single face.
Input
•	face_crop: Cropped face image
Process
•	Calls estimate_gaze
•	Calls estimate_eye_openness
•	Converts gaze into engagement factors
Formula
•	forward_gaze = 1 if gaze is forward else 0.25
•	head_pose = 1.0 if forward else 0.3
•	score = (forward_gaze x 40) + (eye_openness x 0.25) + (head_pose x 35)
Output
•	Dictionary with
o	score (0-100)
o	gaze direction
o	eye_openness value

### get_zone(bbox, frame_w, frame_h)
This function assigns each detected face to a spatial grid zone.
Input
•	bbox: (x, y, width, height)
•	frame_w: frame width
•	frame_h: frame height
Process
•	Computes center of face bounding box
•	Maps center position to grid indices
•	Grid size is 4 rows x 6 columns
Logic
•	Row index based on vertical position
•	Column index based on horizontal position
•	Values are clamped within valid grid limits
Output
•	Zone ID string like R2C3

### detect_faces(frame)
This function detects faces in a frame.
Input
•	frame: full video frame
Process
•	Downscales frame using DETECT_SCALE for faster processing
•	Converts to grayscale
•	Applies CLAHE for contrast enhancement
•	Uses Haar Cascade classifier to detect faces
•	Rescales bounding boxes back to original frame size
•	Extracts face crops
Output
•	List of dictionaries
o	bbox: bounding box
o	face_crop: cropped face image

### process_window(frames_in_window, frame_w, frame_h)
This function processes a batch of frames within a time window.
Input
•	frames_in_window: list of sampled frames
•	frame_w, frame_h: frame dimensions
Process
•	For each frame
o	Detect faces
o	Score each face
o	Assign zone
•	Aggregates results across the window
Computed Metrics
•	Average engagement score
•	Total persons detected
•	Gaze distribution counts
•	Average engagement per spatial zone
•	Stores a few face crops for visualization
Output
•	Dictionary containing
o	engagement_score
o	persons_count
o	gaze_distribution
o	spatial_zones
o	face_crops

### make_collage_b64(face_crops, cell=60)
This function creates a collage of face images.
Input
•	face_crops: list of cropped face images
•	cell: size of each image
Process
•	Resizes each face to fixed size
•	Stacks images horizontally
•	Encodes result as JPEG
•	Converts to base64 string
Output
•	Base64 encoded image string
•	Empty string if no faces

### generate_engagement_report(windows, stats, output_path)
This function generates an HTML report.
Input
•	windows: list of window-level results
•	stats: overall session statistics
•	output_path: file path for HTML
Process
•	Builds a color scale for heatmap
•	Creates heatmap table (4x6 grid)
•	Builds engagement timeline using bars
•	Displays worst 3 windows with face collages
Output
•	Saves a self-contained HTML report

## Output Files
engagement_output.json
•	Contains structured session data including
o	overall engagement
o	per-window results
o	spatial heatmap
o	worst windows
engagement_report.html
•	Visual report including
o	summary
o	engagement timeline
o	classroom heatmap
o	worst performing segments


