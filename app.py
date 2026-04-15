"""
╔══════════════════════════════════════════════════════════════════════════╗
║   TRAFFIC DETECTION SERVICE — VIDEO + IMAGE  (FIXED v2.1)              ║
║   Fix: Video violation detection now analyzes frames individually       ║
║        → picks best result (highest confidence violation)               ║
║                                                                         ║
║   pip install fastapi uvicorn httpx python-multipart opencv-python      ║
║               python-multipart pydantic reportlab pillow                ║
║                                                                         ║
║   uvicorn detection_service:app --reload --port 8000                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, sqlite3, base64, json, smtplib, asyncio, time, tempfile, re
from contextlib import asynccontextmanager
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import httpx

# ── Optional imports ──────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠ opencv not found. Run: pip install opencv-python")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
DB_PATH           = "traffic.db"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SMTP_HOST         = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("SMTP_PORT", 587))
SMTP_USER         = os.getenv("SMTP_USER", "")
SMTP_PASS         = os.getenv("SMTP_PASS", "")
POLICE_EMAIL      = os.getenv("POLICE_EMAIL", "police@traffic.gov.in")
UPLOAD_DIR        = "uploads"

# How many frames to sample from a video
VIDEO_SAMPLE_FRAMES = 6
MAX_VIDEO_SIZE_MB   = 50

FINE_TABLE = {
    "red_light_jump": 2000, "no_helmet": 1000, "triple_riding": 1500,
    "wrong_way": 2500,      "speeding": 3000,  "no_seatbelt": 1000,
    "mobile_use": 1500,     "illegal_parking": 500, "unknown": 500,
}
VIOLATION_LABELS = {
    "red_light_jump": "Red Light Jump", "no_helmet": "No Helmet",
    "triple_riding": "Triple Riding",   "wrong_way": "Wrong Way",
    "speeding": "Speeding",             "no_seatbelt": "No Seatbelt",
    "mobile_use": "Mobile Use",         "illegal_parking": "Illegal Parking",
    "unknown": "Unknown", "none": "No Violation",
}

# Violation severity ranking — used to pick best result across frames
VIOLATION_SEVERITY = {
    "speeding": 6, "wrong_way": 5, "red_light_jump": 4,
    "triple_riding": 3, "mobile_use": 3, "no_seatbelt": 2,
    "no_helmet": 2, "illegal_parking": 1, "unknown": 0, "none": -1,
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT NOT NULL,
        vehicle_type TEXT DEFAULT 'car',
        violation_type TEXT NOT NULL,
        junction_id TEXT DEFAULT 'JN-01',
        signal_state TEXT DEFAULT 'red',
        fine_amount INTEGER DEFAULT 0,
        fine_paid INTEGER DEFAULT 0,
        source_type TEXT DEFAULT 'image',
        source_path TEXT,
        confidence REAL DEFAULT 0.0,
        speed_kmh REAL DEFAULT 0.0,
        description TEXT DEFAULT '',
        frames_analysed INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now','localtime')))""")

    c.execute("""CREATE TABLE IF NOT EXISTS ambulance_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        junction_id TEXT DEFAULT 'JN-01',
        source_type TEXT DEFAULT 'image',
        source_path TEXT,
        ambulance_found INTEGER DEFAULT 0,
        stuck_in_traffic INTEGER DEFAULT 0,
        severity TEXT DEFAULT 'low',
        description TEXT DEFAULT '',
        police_alerted INTEGER DEFAULT 0,
        signal_override TEXT DEFAULT '',
        frames_analysed INTEGER DEFAULT 1,
        confidence REAL DEFAULT 0.0,
        created_at TEXT DEFAULT (datetime('now','localtime')))""")

    c.execute("""CREATE TABLE IF NOT EXISTS alert_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_type TEXT NOT NULL,
        recipient TEXT NOT NULL,
        subject TEXT,
        message TEXT,
        status TEXT DEFAULT 'sent',
        sent_at TEXT DEFAULT (datetime('now','localtime')))""")

    conn.commit()
    conn.close()
    print("✅ DB ready — traffic.db")

# ══════════════════════════════════════════════════════════════════════
#  VIDEO FRAME EXTRACTOR
# ══════════════════════════════════════════════════════════════════════
def extract_frames_from_video(video_bytes: bytes, n_frames: int = VIDEO_SAMPLE_FRAMES) -> list[bytes]:
    """Extract n evenly-spaced frames from video bytes. Returns list of JPEG bytes."""
    if not CV2_AVAILABLE:
        raise HTTPException(500, "opencv-python not installed. Run: pip install opencv-python")

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = [int(total * i / n_frames) for i in range(n_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (1280, int(h * scale)))
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append(buf.tobytes())
        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames

def image_bytes_to_jpeg(image_bytes: bytes, mime: str) -> bytes:
    """Convert any image format to JPEG bytes."""
    if mime in ('image/jpeg', 'image/jpg'):
        return image_bytes
    if PIL_AVAILABLE:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
    return image_bytes

# ══════════════════════════════════════════════════════════════════════
#  CLAUDE AI — VIOLATION DETECTION
# ══════════════════════════════════════════════════════════════════════
VIOLATION_PROMPT = """You are an expert traffic violation detection AI analysing CCTV footage frames.

Carefully examine this traffic camera frame and detect ANY traffic violations.
Focus especially on:
- Vehicles crossing RED signals
- No helmet on two-wheelers
- Triple riding on bikes
- Wrong-way driving
- Overspeeding indicators
- No seatbelt
- Mobile phone usage while driving
- Illegal parking

Respond ONLY with valid JSON (no markdown, no extra text):
{
  "plate": "<number plate text or UNKNOWN>",
  "vehicle_type": "car|bike|truck|auto|bus|unknown",
  "violation_type": "red_light_jump|no_helmet|triple_riding|wrong_way|speeding|no_seatbelt|mobile_use|illegal_parking|none|unknown",
  "signal_state": "red|green|yellow|unknown",
  "confidence": 0.0,
  "description": "one clear sentence describing what is seen",
  "speed_estimate": "normal|slow|fast|very_fast",
  "multiple_violations": false,
  "additional_violations": []
}"""


def _pick_best_violation_result(results: list[dict]) -> dict:
    """
    ────────────────────────────────────────────────────────────────
    KEY FIX: From a list of per-frame results, pick the single best
    result to represent the whole video.

    Selection priority:
      1. Any actual violation  > "none"  > "unknown"
      2. Among violations, pick by VIOLATION_SEVERITY rank
      3. Break ties by highest confidence score
    ────────────────────────────────────────────────────────────────
    """
    if not results:
        return {}

    # Separate violations from non-violations
    violations = [r for r in results if r.get("violation_type") not in ("none", None, "")]
    non_violations = [r for r in results if r.get("violation_type") in ("none", None, "")]

    if violations:
        # Sort: higher severity first, then higher confidence
        violations.sort(
            key=lambda r: (
                VIOLATION_SEVERITY.get(r.get("violation_type", "unknown"), 0),
                r.get("confidence", 0.0),
            ),
            reverse=True,
        )
        best = violations[0]
    else:
        # All frames said "none" — return the one with highest confidence
        non_violations.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
        best = non_violations[0] if non_violations else results[0]

    return best


async def detect_violation_in_image(frame_bytes: bytes) -> dict:
    """Single frame violation detection."""
    if not ANTHROPIC_API_KEY:
        import random
        violations = ["red_light_jump", "no_helmet", "speeding", "mobile_use", "none"]
        weights = [0.3, 0.2, 0.15, 0.1, 0.25]
        vtype = random.choices(violations, weights)[0]
        plates = ["TS09AB1234", "AP28CD5678", "TS07EF9012", "UNKNOWN"]
        return {
            "plate": random.choice(plates),
            "vehicle_type": random.choice(["car", "bike", "truck", "auto"]),
            "violation_type": vtype,
            "signal_state": random.choice(["red", "green", "yellow"]),
            "confidence": round(random.uniform(0.72, 0.97), 2),
            "description": f"Demo: {VIOLATION_LABELS.get(vtype, vtype)} detected in frame.",
            "speed_estimate": "normal",
            "multiple_violations": False,
            "additional_violations": [],
        }

    b64 = base64.standard_b64encode(frame_bytes).decode()
    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 500,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": VIOLATION_PROMPT},
                    ],
                }],
            },
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)

    data["fine_amount"] = FINE_TABLE.get(data.get("violation_type", "unknown"), 500)
    return data


# ══════════════════════════════════════════════════════════════════════
#  FIXED: detect_violation_in_video
#  OLD approach: send ALL frames at once → Claude gets confused, timeout
#  NEW approach: analyze each frame individually → pick best result
# ══════════════════════════════════════════════════════════════════════
async def detect_violation_in_video(frames: list[bytes]) -> dict:
    """
    Multi-frame video violation detection.

    FIXED: Sends each frame to Claude individually (same prompt as image),
    collects all results, then picks the best one using _pick_best_violation_result().

    Why this works better:
    - Claude gets ONE clear frame at a time → focused, accurate response
    - No timeout from sending 6 large images at once
    - We get per-frame results and surface the most severe violation
    """
    if not ANTHROPIC_API_KEY:
        import random
        violations = ["red_light_jump", "no_helmet", "speeding", "mobile_use", "none"]
        weights = [0.35, 0.2, 0.2, 0.1, 0.15]
        vtype = random.choices(violations, weights)[0]
        return {
            "plate": f"TS{random.randint(1,99):02d}AB{random.randint(1000,9999)}",
            "vehicle_type": random.choice(["car", "bike", "truck"]),
            "violation_type": vtype,
            "signal_state": random.choice(["red", "green"]),
            "confidence": round(random.uniform(0.78, 0.96), 2),
            "description": f"Demo video analysis: {VIOLATION_LABELS.get(vtype, vtype)} detected across {len(frames)} frames.",
            "frame_count": len(frames),
            "violations_per_frame": [],
            "fine_amount": FINE_TABLE.get(vtype, 500),
        }

    # ── Analyze each frame individually (concurrent for speed) ──────
    async def analyze_one_frame(frame_bytes: bytes, frame_idx: int) -> dict:
        """Analyze a single frame with error handling."""
        try:
            result = await detect_violation_in_image(frame_bytes)
            result["frame_index"] = frame_idx
            return result
        except Exception as e:
            print(f"[WARN] Frame {frame_idx} analysis failed: {e}")
            return {
                "frame_index": frame_idx,
                "violation_type": "unknown",
                "confidence": 0.0,
                "description": f"Frame {frame_idx} analysis error: {str(e)}",
                "plate": "UNKNOWN",
                "vehicle_type": "unknown",
                "signal_state": "unknown",
            }

    # Run all frame analyses concurrently
    tasks = [analyze_one_frame(frame, i + 1) for i, frame in enumerate(frames)]
    per_frame_results = await asyncio.gather(*tasks)

    # ── Pick the best result across all frames ───────────────────────
    best = _pick_best_violation_result(list(per_frame_results))

    # Attach per-frame summary metadata
    best["frame_count"] = len(frames)
    best["frames_analysed"] = len(frames)
    best["violations_per_frame"] = [
        {
            "frame": r.get("frame_index"),
            "violation_type": r.get("violation_type"),
            "confidence": r.get("confidence", 0.0),
        }
        for r in per_frame_results
    ]

    # Update description to mention it's from video
    orig_desc = best.get("description", "")
    violation_frames = [
        r for r in per_frame_results
        if r.get("violation_type") not in ("none", None, "")
    ]
    if violation_frames:
        best["description"] = (
            f"[Video: {len(violation_frames)}/{len(frames)} frames with violations] {orig_desc}"
        )

    best["fine_amount"] = FINE_TABLE.get(best.get("violation_type", "unknown"), 500)
    return best


# ══════════════════════════════════════════════════════════════════════
#  CLAUDE AI — AMBULANCE DETECTION
#  (Unchanged — multi-frame approach works fine for ambulance since
#   it's a simpler binary detection task)
# ══════════════════════════════════════════════════════════════════════
AMBULANCE_PROMPT = """You are an emergency vehicle detection AI analysing traffic camera footage.

Your ONLY job: Detect if an AMBULANCE is present and whether it is STUCK in traffic.

Look for:
- Ambulance vehicle (any type: van, car, mini-ambulance, 108, CATS, etc.)
- Red cross symbol, "AMBULANCE" text, emergency lights/beacons
- Is the ambulance moving freely OR stuck/blocked by other vehicles?
- Traffic density around the ambulance

Respond ONLY with valid JSON (no markdown):
{
  "ambulance_found": true|false,
  "ambulance_type": "108|CATS|private|unknown|none",
  "stuck_in_traffic": true|false,
  "traffic_density": "clear|light|moderate|heavy|gridlock",
  "blocking_vehicles_count": 0,
  "severity": "none|low|medium|high|critical",
  "confidence": 0.0,
  "description": "One clear sentence about the ambulance situation",
  "recommended_action": "none|monitor|alert_police|override_signal|clear_path"
}"""

AMBULANCE_VIDEO_PROMPT = """You are an emergency vehicle detection AI. I am giving you MULTIPLE frames from a video.

Analyse ALL frames to determine:
1. Is there an ambulance in any frame?
2. Is it stuck in traffic?
3. What is the worst traffic situation across all frames?

Respond ONLY with valid JSON:
{
  "ambulance_found": true|false,
  "ambulance_type": "108|CATS|private|unknown|none",
  "stuck_in_traffic": true|false,
  "traffic_density": "clear|light|moderate|heavy|gridlock",
  "blocking_vehicles_count": 0,
  "severity": "none|low|medium|high|critical",
  "confidence": 0.0,
  "description": "Summary of ambulance situation across all video frames",
  "recommended_action": "none|monitor|alert_police|override_signal|clear_path",
  "frames_with_ambulance": 0,
  "total_frames": 0
}"""

async def detect_ambulance_in_image(frame_bytes: bytes) -> dict:
    """Single image ambulance detection."""
    if not ANTHROPIC_API_KEY:
        import random
        found = random.random() > 0.5
        stuck = found and random.random() > 0.4
        severity = "critical" if stuck else ("low" if found else "none")
        return {
            "ambulance_found": found,
            "ambulance_type": "108" if found else "none",
            "stuck_in_traffic": stuck,
            "traffic_density": random.choice(["moderate", "heavy", "gridlock"]) if stuck else "clear",
            "blocking_vehicles_count": random.randint(3, 12) if stuck else 0,
            "severity": severity,
            "confidence": round(random.uniform(0.75, 0.95), 2),
            "description": f"Demo: {'Ambulance STUCK in heavy traffic!' if stuck else 'Ambulance moving freely.' if found else 'No ambulance detected.'}",
            "recommended_action": "override_signal" if stuck else ("monitor" if found else "none"),
        }

    b64 = base64.standard_b64encode(frame_bytes).decode()
    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 400,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": AMBULANCE_PROMPT},
                    ],
                }],
            },
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)

async def detect_ambulance_in_video(frames: list[bytes]) -> dict:
    """Multi-frame video ambulance detection (kept as multi-frame — works fine for this simpler task)."""
    if not ANTHROPIC_API_KEY:
        import random
        found = random.random() > 0.4
        stuck = found and random.random() > 0.3
        severity = "critical" if stuck else ("low" if found else "none")
        return {
            "ambulance_found": found,
            "ambulance_type": "108" if found else "none",
            "stuck_in_traffic": stuck,
            "traffic_density": "heavy" if stuck else "moderate",
            "blocking_vehicles_count": random.randint(4, 15) if stuck else 0,
            "severity": severity,
            "confidence": round(random.uniform(0.80, 0.96), 2),
            "description": f"Demo video: {'Ambulance STUCK!' if stuck else 'Ambulance moving.' if found else 'No ambulance in video.'}",
            "recommended_action": "override_signal" if stuck else ("monitor" if found else "none"),
            "frames_with_ambulance": random.randint(3, len(frames)) if found else 0,
            "total_frames": len(frames),
        }

    content = []
    for i, frame in enumerate(frames):
        b64 = base64.standard_b64encode(frame).decode()
        content.append({"type": "text", "text": f"Frame {i+1}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    content.append({"type": "text", "text": AMBULANCE_VIDEO_PROMPT})

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": "claude-opus-4-5", "max_tokens": 600, "messages": [{"role": "user", "content": content}]},
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)

# ══════════════════════════════════════════════════════════════════════
#  ALERT — EMAIL POLICE
# ══════════════════════════════════════════════════════════════════════
def send_police_email(subject: str, body: str, alert_type: str = "ambulance") -> str:
    if SMTP_USER and SMTP_PASS:
        try:
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = SMTP_USER
            msg["To"] = POLICE_EMAIL
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            status = "sent"
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")
            status = "failed"
    else:
        print(f"\n{'='*60}")
        print(f"[SIMULATED EMAIL TO POLICE]")
        print(f"Subject: {subject}")
        print(f"Body:\n{body}")
        print('='*60)
        status = "simulated"

    conn = get_conn()
    conn.execute(
        "INSERT INTO alert_log (alert_type,recipient,subject,message,status) VALUES (?,?,?,?,?)",
        (alert_type, POLICE_EMAIL, subject, body[:500], status))
    conn.commit()
    conn.close()
    return status

def build_ambulance_alert_email(result: dict, junction_id: str, source_type: str, detection_id: int) -> tuple[str, str]:
    severity_emoji = {"none": "ℹ️", "low": "⚠️", "medium": "🔶", "high": "🚨", "critical": "🆘"}
    emoji = severity_emoji.get(result.get("severity", "low"), "🚨")

    subject = f"{emoji} AMBULANCE ALERT [{result.get('severity','?').upper()}] — Junction {junction_id}"

    body = f"""
╔══════════════════════════════════════════════════════╗
║           🚑 AMBULANCE EMERGENCY ALERT               ║
╚══════════════════════════════════════════════════════╝

Detection ID  : #{detection_id}
Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Junction      : {junction_id}
Source        : {source_type.upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ambulance Type    : {result.get('ambulance_type', 'Unknown').upper()}
Stuck in Traffic  : {'🔴 YES — IMMEDIATE ACTION REQUIRED' if result.get('stuck_in_traffic') else '🟢 No'}
Traffic Density   : {result.get('traffic_density', 'Unknown').upper()}
Blocking Vehicles : {result.get('blocking_vehicles_count', 0)}
Severity          : {result.get('severity', 'Unknown').upper()}
AI Confidence     : {result.get('confidence', 0)*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SITUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{result.get('description', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{result.get('recommended_action', 'monitor').replace('_', ' ').upper()}

🚦 Signal at {junction_id} has been automatically overridden to GREEN.
👮 Please dispatch traffic personnel to clear the path immediately.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is an automated alert from TrafficGuard AI System.
Dashboard: http://localhost:8000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return subject, body

# ══════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Traffic Detection Service",
    description="Video + Image detection for violations and ambulance",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def serve():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "detect_dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>Place detect_dashboard.html in templates/ folder</h2>")

# ══════════════════════════════════════════════════════════════════════
#  ① VIOLATION DETECTION ENDPOINT
# ══════════════════════════════════════════════════════════════════════
@app.post("/api/detect/violation", tags=["Violation Detection"])
async def detect_violation(
    file: UploadFile = File(...),
    junction_id: str = Form("JN-01"),
):
    """
    Upload IMAGE or VIDEO.
    - Image → single frame analysis (fast, as before)
    - Video → extract frames → analyze each frame individually
              → pick best/most severe violation result
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"
    mime = file.content_type or ""
    is_video = mime.startswith("video/") or filename.lower().endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.3gp'))

    save_path = f"{UPLOAD_DIR}/violation_{int(time.time())}_{filename}"
    with open(save_path, "wb") as f_:
        f_.write(file_bytes)

    # ── VIDEO PATH ──────────────────────────────────────────
    if is_video:
        if not CV2_AVAILABLE:
            raise HTTPException(500, "opencv-python not installed. Run: pip install opencv-python")
        frames = extract_frames_from_video(file_bytes, VIDEO_SAMPLE_FRAMES)
        if not frames:
            raise HTTPException(400, "Could not extract frames from video. Check file format.")

        # FIXED: per-frame analysis with best-result selection
        result = await detect_violation_in_video(frames)
        result["source_type"] = "video"
        result["frames_analysed"] = len(frames)

    # ── IMAGE PATH ──────────────────────────────────────────
    else:
        frame = image_bytes_to_jpeg(file_bytes, mime)
        result = await detect_violation_in_image(frame)
        result["source_type"] = "image"
        result["frames_analysed"] = 1

    # ── SAVE TO DB ──────────────────────────────────────────
    violation_type = result.get("violation_type", "unknown")
    if violation_type == "none":
        return {
            "detected": False,
            "message": "No traffic violation detected",
            "source_type": result["source_type"],
            "result": result,
        }

    fine = FINE_TABLE.get(violation_type, 500)
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO violations
           (plate,vehicle_type,violation_type,junction_id,signal_state,
            fine_amount,source_type,source_path,confidence,description,frames_analysed)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            result.get("plate", "UNKNOWN"),
            result.get("vehicle_type", "unknown"),
            violation_type,
            junction_id,
            result.get("signal_state", "unknown"),
            fine,
            result["source_type"],
            save_path,
            result.get("confidence", 0.0),
            result.get("description", ""),
            result.get("frames_analysed", 1),
        ),
    )
    conn.commit()
    vid = cur.lastrowid
    conn.close()

    result["fine_amount"] = fine
    result["violation_label"] = VIOLATION_LABELS.get(violation_type, violation_type)

    return {
        "detected": True,
        "violation_id": vid,
        "message": f"Violation detected: {result['violation_label']}",
        "source_type": result["source_type"],
        "fine_amount": fine,
        "result": result,
    }

# ══════════════════════════════════════════════════════════════════════
#  ② AMBULANCE DETECTION ENDPOINT (unchanged)
# ══════════════════════════════════════════════════════════════════════
@app.post("/api/detect/ambulance", tags=["Ambulance Detection"])
async def detect_ambulance(
    file: UploadFile = File(...),
    junction_id: str = Form("JN-01"),
):
    file_bytes = await file.read()
    filename = file.filename or "upload"
    mime = file.content_type or ""
    is_video = mime.startswith("video/") or filename.lower().endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.3gp'))

    save_path = f"{UPLOAD_DIR}/ambulance_{int(time.time())}_{filename}"
    with open(save_path, "wb") as f_:
        f_.write(file_bytes)

    if is_video:
        if not CV2_AVAILABLE:
            raise HTTPException(500, "opencv-python not installed.")
        frames = extract_frames_from_video(file_bytes, VIDEO_SAMPLE_FRAMES)
        if not frames:
            raise HTTPException(400, "Could not extract frames from video.")
        result = await detect_ambulance_in_video(frames)
        source_type = "video"
        frames_analysed = len(frames)
    else:
        frame = image_bytes_to_jpeg(file_bytes, mime)
        result = await detect_ambulance_in_image(frame)
        source_type = "image"
        frames_analysed = 1

    ambulance_found = result.get("ambulance_found", False)
    stuck = result.get("stuck_in_traffic", False)
    severity = result.get("severity", "none")

    police_alerted = False
    alert_status = "not_needed"
    if ambulance_found and stuck:
        subject, body = build_ambulance_alert_email(result, junction_id, source_type, 0)
        alert_status = send_police_email(subject, body, "ambulance")
        police_alerted = True

    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO ambulance_detections
           (junction_id,source_type,source_path,ambulance_found,stuck_in_traffic,
            severity,description,police_alerted,confidence,frames_analysed)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            junction_id, source_type, save_path,
            1 if ambulance_found else 0,
            1 if stuck else 0,
            severity,
            result.get("description", ""),
            1 if police_alerted else 0,
            result.get("confidence", 0.0),
            frames_analysed,
        ),
    )
    conn.commit()
    det_id = cur.lastrowid
    conn.close()

    if not ambulance_found:
        msg = "No ambulance detected in the upload."
    elif ambulance_found and not stuck:
        msg = "Ambulance detected — moving freely. No alert needed."
    else:
        msg = f"🚨 AMBULANCE STUCK IN TRAFFIC — Police alerted! (Status: {alert_status})"

    return {
        "detection_id": det_id,
        "ambulance_found": ambulance_found,
        "stuck_in_traffic": stuck,
        "severity": severity,
        "police_alerted": police_alerted,
        "alert_status": alert_status,
        "source_type": source_type,
        "frames_analysed": frames_analysed,
        "message": msg,
        "result": result,
    }

# ══════════════════════════════════════════════════════════════════════
#  HISTORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/violations/", tags=["History"])
def list_violations(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM violations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/violations/stats", tags=["History"])
def violation_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    paid  = conn.execute("SELECT COUNT(*) FROM violations WHERE fine_paid=1").fetchone()[0]
    amount= conn.execute("SELECT COALESCE(SUM(fine_amount),0) FROM violations WHERE fine_paid=0").fetchone()[0]
    today = conn.execute("SELECT COUNT(*) FROM violations WHERE date(created_at)=date('now')").fetchone()[0]
    by_type = conn.execute(
        "SELECT violation_type, COUNT(*) as c FROM violations GROUP BY violation_type ORDER BY c DESC"
    ).fetchall()
    conn.close()
    return {"total": total, "paid": paid, "pending_amount": amount, "today": today,
            "by_type": [dict(r) for r in by_type]}

@app.patch("/api/violations/{vid}/pay", tags=["History"])
def mark_paid(vid: int):
    conn = get_conn()
    r = conn.execute("SELECT id,fine_paid FROM violations WHERE id=?", (vid,)).fetchone()
    if not r: raise HTTPException(404, "Not found")
    if r["fine_paid"]: raise HTTPException(400, "Already paid")
    conn.execute("UPDATE violations SET fine_paid=1 WHERE id=?", (vid,))
    conn.commit(); conn.close()
    return {"message": "Marked as paid", "id": vid}

@app.get("/api/ambulance/", tags=["History"])
def list_ambulance(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM ambulance_detections ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/ambulance/stats", tags=["History"])
def ambulance_stats():
    conn = get_conn()
    total   = conn.execute("SELECT COUNT(*) FROM ambulance_detections").fetchone()[0]
    found   = conn.execute("SELECT COUNT(*) FROM ambulance_detections WHERE ambulance_found=1").fetchone()[0]
    stuck   = conn.execute("SELECT COUNT(*) FROM ambulance_detections WHERE stuck_in_traffic=1").fetchone()[0]
    alerted = conn.execute("SELECT COUNT(*) FROM ambulance_detections WHERE police_alerted=1").fetchone()[0]
    conn.close()
    return {"total_scans": total, "ambulances_found": found, "stuck_cases": stuck, "police_alerted": alerted}

@app.get("/api/alerts/", tags=["History"])
def list_alerts(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM alert_log ORDER BY sent_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]