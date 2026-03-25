"""
classroom_engagement.py
Sentio Mind · Project 3 · Classroom Engagement Heatmap & Group Analysis

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
"""

import cv2
import json
import base64
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path("video_sample_1.mov")
REPORT_HTML_OUT = Path("engagement_report.html")
OUTPUT_JSON     = Path("engagement_output.json")

WINDOW_SEC      = 6       # seconds per analysis window — keep this configurable
SAMPLE_EVERY_N  = 5       # analyse every Nth frame inside a window (speed trade-off)
GRID_ROWS       = 4
GRID_COLS       = 6
DETECT_SCALE    = 0.5     # downscale factor for face detection to speed up

# ---------------------------------------------------------------------------
# GAZE & EYE OPENNESS
# ---------------------------------------------------------------------------

def estimate_gaze(face_crop: np.ndarray) -> str:
    """
    Return one of: "forward", "down", "left", "right", "unknown"

    Preferred: MediaPipe Face Mesh iris landmarks — compute offset of iris
    centre from eye centre, normalise to eye width. Threshold at 0.1.

    Fallback: brightness gradient across the eye region — if the dark region
    (iris) is left of centre, gaze is "left" etc.

    TODO: implement
    """

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    eye_region = gray[int(h*0.2):int(h*0.5), :]
    if eye_region.size == 0:
        return "unknown"
    left = np.mean(eye_region[:, :w//2])
    right = np.mean(eye_region[:, w//2:])
    diff = left - right
    if abs(diff) < 5:
        return "forward"
    elif diff > 5:
        return "left"
    elif diff < -5:
        return "right"
    return "unknown"

def estimate_eye_openness(face_crop: np.ndarray) -> float:
    """
    Eye height / eye width ratio from face landmarks, scaled 0–100.
    Return 50.0 if no landmark found.
    TODO: implement
    """

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    eye_region = gray[int(h*0.2):int(h*0.5), :]
    if eye_region.size == 0:
        return 50.0
    variance = np.var(eye_region)
    # Normalize roughly into 0 to 100
    score = np.clip(variance / 50, 0, 100)
    return float(score)

# ---------------------------------------------------------------------------
# PER-FACE ENGAGEMENT SCORE
# ---------------------------------------------------------------------------

def score_face(face_crop: np.ndarray) -> dict:
    """
    Compute engagement for one face.
    Return: { "score": float, "gaze": str, "eye_openness": float }

    Formula from README:
      face_engagement = (forward_gaze × 40) + (eye_openness × 0.25) + (head_pose × 0.35)
      forward_gaze = 1 if gaze == "forward" else 0.25
      head_pose    = 1.0 if gaze == "forward" else 0.3  (simple proxy)

    TODO: implement
    """

    gaze = estimate_gaze(face_crop)
    eye_openness = estimate_eye_openness(face_crop)
    forward_gaze = 1 if gaze == "forward" else 0.25
    head_pose = 1.0 if gaze == "forward" else 0.3
    score = (forward_gaze * 40) + (eye_openness * 0.25) + (head_pose * 35)
    return {"score": float(np.clip(score, 0, 100)), "gaze": gaze, "eye_openness": eye_openness}

# ---------------------------------------------------------------------------
# SPATIAL ZONE
# ---------------------------------------------------------------------------

def get_zone(bbox: tuple, frame_w: int, frame_h: int) -> str:
    """
    bbox = (x, y, w, h) in pixels. Use face centre.
    Return zone ID string like "R2C3".
    Clamp column 0–5, row 0–3.
    TODO: implement the formula from README
    """

    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    col = int((cx / frame_w) * GRID_COLS)
    row = int((cy / frame_h) * GRID_ROWS)
    col = max(0, min(GRID_COLS - 1, col))
    row = max(0, min(GRID_ROWS - 1, row))
    return f"R{row+1}C{col+1}"

# ---------------------------------------------------------------------------
# FACE DETECTION IN ONE FRAME
# ---------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame: np.ndarray) -> list:
    """
    Detect all faces. Return: [{"bbox": (x,y,w,h), "face_crop": ndarray}, ...]
    Apply CLAHE on the frame first.
    Use Haar cascade or MediaPipe Face Detection — whichever works better on your data.
    TODO: implement
    """

    detections = []
    small_frame = cv2.resize(frame, (0,0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        # scale back to original frame
        x = int(x / DETECT_SCALE)
        y = int(y / DETECT_SCALE)
        w = int(w / DETECT_SCALE)
        h = int(h / DETECT_SCALE)
        crop = frame[y:y+h, x:x+w]
        detections.append({"bbox": (x, y, w, h), "face_crop": crop})
    return detections

# ---------------------------------------------------------------------------
# PROCESS ONE TIME WINDOW
# ---------------------------------------------------------------------------

def process_window(frames_in_window: list, frame_w: int, frame_h: int) -> dict:
    """
    frames_in_window: [(frame_idx, timestamp_sec, ndarray), ...]

    For each sampled frame:
      - detect faces
      - score each face
      - assign each face to a spatial zone

    Aggregate:
      - group engagement_score = mean of all face scores
      - gaze_distribution count
      - spatial_zones = mean score per zone

    Return a dict that becomes one entry in time_windows.
    Also attach face_crops list (a few small crops for evidence thumbnails — not in JSON).

    TODO: implement
    """

    scores = []
    gaze_dist = defaultdict(int)
    zone_scores = defaultdict(list)
    face_crops = []
    for _, _, frame in frames_in_window:
        detections = detect_faces(frame)
        for det in detections:
            result = score_face(det["face_crop"])
            scores.append(result["score"])
            gaze_dist[result["gaze"]] += 1
            zone = get_zone(det["bbox"], frame_w, frame_h)
            zone_scores[zone].append(result["score"])
            if len(face_crops) < 10:
                face_crops.append(det["face_crop"])
    avg_score = int(np.mean(scores)) if scores else 0
    spatial_avg = {z: int(np.mean(v)) for z, v in zone_scores.items()}
    return {
        "engagement_score": avg_score,
        "persons_count": len(scores),
        "gaze_distribution": {
            "forward": gaze_dist["forward"],
            "down": gaze_dist["down"],
            "left": gaze_dist["left"],
            "right": gaze_dist["right"],
            "unknown": gaze_dist["unknown"],
        },
        "spatial_zones": spatial_avg,
        "face_crops": face_crops,
    }

# ---------------------------------------------------------------------------
# COLLAGE HELPER
# ---------------------------------------------------------------------------

def make_collage_b64(face_crops: list, cell: int = 60) -> str:
    """
    Stack face crops horizontally into a small collage, return as base64 JPEG.
    Resize each crop to cell × cell before stacking.
    Return empty string if no crops.
    TODO: implement
    """

    if not face_crops:
        return ""
    resized = [cv2.resize(f, (cell, cell)) for f in face_crops]
    collage = np.hstack(resized)
    _, buffer = cv2.imencode(".jpg", collage)
    return base64.b64encode(buffer).decode("utf-8")

# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_engagement_report(windows: list, stats: dict, output_path: Path):
    """
    Write engagement_report.html — self-contained, no CDN.

    Must include:
      1. Session summary numbers
      2. Engagement timeline chart  (use Chart.js bundled inline OR plain HTML bars)
      3. 4×6 heatmap: HTML table, cell background-color based on zone score
         colour scale: red (#ef4444) at 0, white at 50, green (#22c55e) at 100
      4. 3 worst windows: timestamp + score + face collage image

    For Chart.js offline: download chart.umd.min.js and paste between <script> tags.
    TODO: implement
    """
    
    def color(score):
        if score < 50:
            return f"rgb(239,{int(255*(score/50))},{int(255*(score/50))})"
        else:
            return f"rgb({int(255*(1-(score-50)/50))},255,{int(255*(1-(score-50)/50))})"
    heatmap_html = ""
    for r in range(GRID_ROWS):
        heatmap_html += "<tr>"
        for c in range(GRID_COLS):
            zid = f"R{r+1}C{c+1}"
            val = stats["session_spatial_heatmap"][zid]
            heatmap_html += f"<td style='background:{color(val)}'>{val}</td>"
        heatmap_html += "</tr>"
    timeline = "".join(
        f"<div style='width:10px;height:{w['engagement_score']}px;background:#22c55e;margin:1px;display:inline-block'></div>"
        for w in windows
    )
    worst_html = ""
    for w in stats["worst_3_windows"]:
        img = w.get("thumbnail_collage_b64", "")
        worst_html += f"""
        <div>
            <p>Window {w['window_id']} ({w['start_sec']}s) - {w['score']}%</p>
            <img src="data:image/jpeg;base64,{img}" />
        </div>
        """
    html = f"""
    <html>
    <body>
    <h1>Classroom Engagement Report</h1>
    <h2>Summary</h2>
    <p>Overall Engagement: {stats['overall_engagement_score']}%</p>
    <h2>Timeline</h2>
    <div style="display:flex;align-items:flex-end">{timeline}</div>
    <h2>Heatmap</h2>
    <table border="1">{heatmap_html}</table>
    <h2>Worst Windows</h2>
    {worst_html}
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(html)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cap   = cv2.VideoCapture(str(VIDEO_PATH))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps

    frames_per_window = max(1, int(WINDOW_SEC * fps))
    windows     = []
    window_buf  = []
    w_start_sec = 0.0
    frame_idx   = 0

    print(f"Processing {VIDEO_PATH}  |  {dur:.1f}s  |  {total} frames  |  window={WINDOW_SEC}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        print(f"Frame {frame_idx}/{total}")

        if frame_idx % SAMPLE_EVERY_N == 0:
            window_buf.append((frame_idx, ts, frame.copy()))

        if (frame_idx + 1) % frames_per_window == 0 or frame_idx == total - 1:
            result = process_window(window_buf, fw, fh)
            result.update({
                "window_id": len(windows) + 1,
                "start_sec": round(w_start_sec, 2),
                "end_sec":   round(ts, 2),
            })
            windows.append(result)
            window_buf  = []
            w_start_sec = ts + 1 / fps

        frame_idx += 1
    cap.release()

    # Session heatmap
    zone_scores = defaultdict(list)
    for w in windows:
        for zid, score in w["spatial_zones"].items():
            zone_scores[zid].append(score)
    heatmap = {
        f"R{r+1}C{c+1}": int(np.mean(zone_scores[f"R{r+1}C{c+1}"])) if zone_scores[f"R{r+1}C{c+1}"] else 0
        for r in range(GRID_ROWS) for c in range(GRID_COLS)
    }

    scores   = [w["engagement_score"] for w in windows]
    peak     = max(windows, key=lambda w: w["engagement_score"])
    trough   = min(windows, key=lambda w: w["engagement_score"])
    worst3   = sorted(windows, key=lambda w: w["engagement_score"])[:3]

    for w in worst3:
        w["thumbnail_collage_b64"] = make_collage_b64(w.get("face_crops", []))

    def clean(w):
        return {k: v for k, v in w.items() if k != "face_crops"}

    stats = {
        "source":                   "p3_classroom_engagement",
        "video":                    str(VIDEO_PATH),
        "date":                     str(date.today()),
        "session_duration_sec":     round(dur, 2),
        "window_size_sec":          WINDOW_SEC,
        "total_persons_detected":   sum(w["persons_count"] for w in windows),
        "overall_engagement_score": int(np.mean(scores)) if scores else 0,
        "peak_window":    clean(peak),
        "trough_window":  clean(trough),
        "worst_3_windows": [
            {"window_id": w["window_id"], "start_sec": w["start_sec"],
             "end_sec": w["end_sec"], "score": w["engagement_score"],
             "thumbnail_collage_b64": w.get("thumbnail_collage_b64", "")}
            for w in worst3
        ],
        "time_windows":             [clean(w) for w in windows],
        "session_spatial_heatmap":  heatmap,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    generate_engagement_report(windows, stats, REPORT_HTML_OUT)

    print()
    print("=" * 50)
    print(f"  Overall engagement:  {stats['overall_engagement_score']}%")
    print(f"  Peak:    window {peak['window_id']}  @ {peak['start_sec']:.0f}s  =  {peak['engagement_score']}%")
    print(f"  Trough:  window {trough['window_id']}  @ {trough['start_sec']:.0f}s  =  {trough['engagement_score']}%")
    print(f"  Report → {REPORT_HTML_OUT}")
    print(f"  JSON   → {OUTPUT_JSON}")
    print("=" * 50)
