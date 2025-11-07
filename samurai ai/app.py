from flask import Flask, render_template, redirect, url_for, session, request, jsonify, g
from flask_login import login_required, current_user
from msal import ConfidentialClientApplication
from google_auth_oauthlib.flow import Flow
from dotenv import load_dotenv
from datetime import datetime, date, time, timedelta
from functools import wraps
import math
import os
import requests
import sqlite3
import json
import numpy as np
from collections import defaultdict

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# ================= HUGGING FACE AI INTEGRATION =================
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")  # Optional - works without key but with rate limits
HF_API_URL = "https://api-inference.huggingface.co/models/"

# Using multiple models for different tasks
HF_MODELS = {
    "text_generation": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Powerful free model
    "time_prediction": "facebook/opt-1.3b",  # Fast for numerical predictions
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english"
}

def query_huggingface(model_name, payload, max_retries=3):
    """Query Hugging Face API with retry logic"""
    headers = {}
    if HUGGINGFACE_API_KEY:
        headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
    
    url = HF_API_URL + model_name
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 503:  # Model loading
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"HF API Error: {e}")
                return None
            time.sleep(1)
    return None

# ================= ML-BASED STUDY TIME PREDICTOR =================
class StudyTimePredictor:
    """ML-based predictor using historical data and assignment characteristics"""
    
    @staticmethod
    def extract_features(title, class_name, user_history):
        """Extract features from assignment for prediction"""
        features = {
            'title_length': len(title),
            'has_numbers': any(c.isdigit() for c in title),
            'keyword_complexity': 0,
            'class_avg_time': 60,
            'user_avg_time': 60,
            'historical_count': len(user_history)
        }
        
        # Keyword complexity scoring
        complex_keywords = ['project', 'essay', 'research', 'presentation', 'report', 'analysis']
        medium_keywords = ['homework', 'assignment', 'worksheet', 'practice', 'review']
        simple_keywords = ['quiz', 'reading', 'exercise', 'problem']
        
        title_lower = title.lower()
        if any(kw in title_lower for kw in complex_keywords):
            features['keyword_complexity'] = 3
        elif any(kw in title_lower for kw in medium_keywords):
            features['keyword_complexity'] = 2
        elif any(kw in title_lower for kw in simple_keywords):
            features['keyword_complexity'] = 1
        
        # Historical averages
        if user_history:
            class_times = [h['time_spent_minutes'] for h in user_history 
                          if h['class_name'] == class_name and h['time_spent_minutes'] > 0]
            if class_times:
                features['class_avg_time'] = np.mean(class_times)
            
            all_times = [h['time_spent_minutes'] for h in user_history 
                        if h['time_spent_minutes'] > 0]
            if all_times:
                features['user_avg_time'] = np.mean(all_times)
        
        return features
    
    @staticmethod
    def predict(features):
        """Predict study time using feature-based heuristics"""
        base_time = features['user_avg_time']
        
        # Adjust based on complexity
        complexity_multiplier = {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.5}
        adjusted_time = base_time * complexity_multiplier.get(features['keyword_complexity'], 1.0)
        
        # Adjust based on title length
        if features['title_length'] > 50:
            adjusted_time *= 1.2
        
        # Use class average if significantly different
        if abs(features['class_avg_time'] - base_time) > 20:
            adjusted_time = (adjusted_time + features['class_avg_time']) / 2
        
        return max(15, min(240, int(adjusted_time)))  # 15-240 minutes range

# ================= ADAPTIVE LEARNING SYSTEM =================
class AdaptiveLearningEngine:
    """Learns from user behavior and improves over time"""
    
    @staticmethod
    def analyze_user_patterns(user_id, conn):
        """Analyze user's study patterns and productivity"""
        completed = conn.execute("""
            SELECT 
                strftime('%H', started_at) as hour,
                strftime('%w', started_at) as day_of_week,
                time_spent_minutes,
                (julianday(datetime('now')) - julianday(started_at)) as days_ago
            FROM assignments
            WHERE user_id = ? AND completed = 1 AND started_at IS NOT NULL
            ORDER BY started_at DESC
            LIMIT 100
        """, (user_id,)).fetchall()
        
        if not completed:
            return None
        
        patterns = {
            'productive_hours': defaultdict(int),
            'productive_days': defaultdict(int),
            'avg_session_length': 0,
            'total_sessions': len(completed),
            'recent_trend': 'stable'
        }
        
        recent_times = []
        for session in completed:
            if session['hour']:
                patterns['productive_hours'][int(session['hour'])] += 1
            if session['day_of_week']:
                patterns['productive_days'][int(session['day_of_week'])] += 1
            if session['time_spent_minutes']:
                recent_times.append(session['time_spent_minutes'])
        
        if recent_times:
            patterns['avg_session_length'] = np.mean(recent_times)
            
            # Trend analysis
            if len(recent_times) >= 5:
                recent_avg = np.mean(recent_times[:5])
                older_avg = np.mean(recent_times[5:]) if len(recent_times) > 5 else recent_avg
                
                if recent_avg > older_avg * 1.2:
                    patterns['recent_trend'] = 'improving'
                elif recent_avg < older_avg * 0.8:
                    patterns['recent_trend'] = 'declining'
        
        # Find top 3 productive hours
        top_hours = sorted(patterns['productive_hours'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        patterns['best_hours'] = [h[0] for h in top_hours] if top_hours else [14, 16, 20]
        
        return patterns

# ================= FLASK APP SETUP =================
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
DATABASE = 'database.db'

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated

def get_db_connection():
    conn = sqlite3.connect("database.db", timeout=5, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()

# ================= AUTH ROUTES (keeping your existing auth) =================
# NOTE: Microsoft Teams integration is available but currently disabled
# To enable: Uncomment the Teams routes below and update environment variables
# This demonstrates the platform's extensibility for enterprise deployments

# Teams Integration Configuration (Currently Disabled for Competition Demo)
"""
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
REDIRECT_URI_TEAMS = "http://localhost:5000/getAToken"
SCOPE = ["User.Read"]

@app.route("/login_teams")
def login_teams():
    '''Microsoft Teams SSO integration - enables institutional deployment'''
    msal_app = ConfidentialClientApplication(
        CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
    )
    auth_url = msal_app.get_authorization_request_url(
        scopes=SCOPE, redirect_uri=REDIRECT_URI_TEAMS
    )
    return redirect(auth_url)

@app.route("/getAToken")
def getAToken():
    '''Handle Microsoft Teams OAuth callback'''
    code = request.args.get("code")
    if not code:
        return "No code returned from Microsoft", 400

    msal_app = ConfidentialClientApplication(
        CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
    )
    result = msal_app.acquire_token_by_authorization_code(
        code, scopes=SCOPE, redirect_uri=REDIRECT_URI_TEAMS
    )

    if "access_token" in result:
        claims = result.get("id_token_claims")
        user_email = claims.get("preferred_username")
        user_name = claims.get("name")

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (user_email,)).fetchone()

        if not user:
            conn.execute(
                "INSERT INTO users (name, email, xp, level, streak) VALUES (?, ?, ?, ?, ?)",
                (user_name, user_email, 0, 1, 0)
            )
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE email = ?", (user_email,)).fetchone()
        
        user_id = user["id"]
        conn.close()

        session["user"] = {"id": user_id, "name": user_name, "email": user_email}
        session["login_provider"] = "teams"
        return redirect(url_for("dashboard"))

    return f"Login failed: {result.get('error_description')}", 400
"""


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = "http://localhost:5000/google_callback"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login_google")
def login_google():
    """Google OAuth login - Primary authentication method"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": [GOOGLE_REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid"
        ]
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    auth_url, state = flow.authorization_url(prompt="consent")
    session["state"] = state
    return redirect(auth_url)

@app.route("/google_callback")
def google_callback():
    """Handle Google OAuth callback and create/login user"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": [GOOGLE_REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid"
        ]
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    
    try:
        flow.fetch_token(authorization_response=request.url)
    except Exception as e:
        print(f"OAuth Error: {e}")
        return "Authentication failed. Please try again.", 400

    credentials = flow.credentials
    
    try:
        user_info = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        ).json()
    except Exception as e:
        print(f"User info fetch error: {e}")
        return "Failed to retrieve user information.", 400

    user_email = user_info.get("email")
    user_name = user_info.get("name")

    if not user_email or not user_name:
        return "Invalid user information received.", 400

    conn = get_db_connection()
    
    try:
        # Check if user exists by email
        user = conn.execute("SELECT * FROM users WHERE email = ?", (user_email,)).fetchone()

        if not user:
            # Create new user
            cur = conn.execute(
                "INSERT INTO users (name, email, xp, level, streak) VALUES (?, ?, ?, ?, ?)",
                (user_name, user_email, 0, 1, 0)
            )
            conn.commit()
            user_id = cur.lastrowid
            print(f"New user created: {user_name} (ID: {user_id})")
        else:
            # Existing user
            user_id = user["id"]
            print(f"Existing user logged in: {user_name} (ID: {user_id})")

        # Set session
        session["user"] = {"id": user_id, "name": user_name, "email": user_email}
        session["login_provider"] = "google"
        session.permanent = True  # Make session persistent
        
        conn.close()
        return redirect(url_for("dashboard"))
        
    except Exception as e:
        conn.close()
        print(f"Database error: {e}")
        return "Database error occurred. Please try again.", 500
    
# ================= DASHBOARD =================
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user"]["id"]
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    classes = conn.execute("SELECT * FROM classes WHERE user_id = ?", (user_id,)).fetchall()
    assignments = conn.execute("""
        SELECT id, class_id, title, due_date, completed, time_spent_minutes
        FROM assignments 
        WHERE user_id = ?
        ORDER BY due_date ASC
    """, (user_id,)).fetchall()
    conn.close()

    today = date.today()
    processed_assignments = []
    for a in assignments:
        if a["due_date"]:
            due = datetime.strptime(a["due_date"], "%Y-%m-%d").date()
            overdue = due < today and not a["completed"]
        else:
            overdue = False
        processed_assignments.append({**dict(a), "overdue": overdue})

    return render_template("dashboard.html", user=user, classes=classes, 
                         assignments=processed_assignments)

@app.route('/quiz')
@login_required
def quiz():
    return render_template('quiz.html')

@app.route("/assignments")
@login_required
def assignments():
    user_id = session["user"]["id"]
    conn = get_db()
    classes = conn.execute("""
        SELECT id, name FROM classes WHERE user_id = ? ORDER BY name
    """, (user_id,)).fetchall()
    all_assignments = conn.execute("""
        SELECT a.id, a.title, a.due_date, a.completed, a.class_id, a.time_spent_minutes
        FROM assignments a WHERE a.user_id = ? ORDER BY a.due_date ASC
    """, (user_id,)).fetchall()
    conn.close()
    return render_template("assignments.html", assignments=all_assignments, 
                         classes=classes, active="classes")

# ================= CLASS MANAGEMENT =================
@app.route("/add_class", methods=["POST"])
@login_required
def add_class():
    if request.is_json:
        data = request.get_json()
        class_name = data.get("name")
    else:
        class_name = request.form.get("name")

    if not class_name:
        return jsonify({"success": False, "error": "name required"}), 400

    user_id = session.get("user", {}).get("id")
    if not user_id:
        return jsonify({"success": False, "error": "Not logged in"}), 403

    conn = get_db_connection()
    conn.execute("INSERT INTO classes (name, user_id) VALUES (?, ?)", (class_name, user_id))
    conn.commit()
    conn.close()

    if request.is_json:
        return jsonify({"success": True})
    else:
        return redirect(url_for("dashboard"))

@app.route("/delete_class/<int:class_id>", methods=["DELETE"])
@login_required
def delete_class(class_id):
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    class_check = conn.execute("SELECT * FROM classes WHERE id = ? AND user_id = ?", 
                               (class_id, user_id)).fetchone()
    if not class_check:
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403

    conn.execute("DELETE FROM assignments WHERE class_id = ?", (class_id,))
    conn.execute("DELETE FROM classes WHERE id = ?", (class_id,))
    conn.commit()
    conn.close()
    return ("", 204)

# ================= ASSIGNMENT MANAGEMENT =================
@app.route("/add_assignment", methods=["POST"])
@login_required
def add_assignment():
    user_id = session["user"]["id"]
    
    if request.is_json:
        data = request.get_json()
        class_id = data.get("class_id")
        title = data.get("title")
        due_date = data.get("due_date")
    else:
        class_id = request.form.get("class_id")
        title = request.form.get("title")
        due_date = request.form.get("due_date")

    if not class_id or not title:
        if request.is_json:
            return jsonify({"success": False, "error": "class_id and title required"}), 400
        else:
            return redirect(url_for("dashboard"))

    started_at = datetime.now().isoformat()

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO assignments (user_id, class_id, title, due_date, completed, started_at, time_spent_minutes)
        VALUES (?, ?, ?, ?, 0, ?, 0)
    """, (user_id, class_id, title, due_date, started_at))
    conn.commit()
    conn.close()

    if request.is_json:
        return jsonify({"success": True})
    else:
        return redirect(url_for("dashboard"))

@app.route("/complete_assignment", methods=["POST"])
@login_required
def complete_assignment():
    user_id = session["user"]["id"]
    data = request.get_json()
    assignment_id = data.get("assignment_id")
    completed = data.get("completed", True)
    time_spent_minutes = data.get("time_spent_minutes", 0)

    if assignment_id is None:
        return jsonify({"success": False, "error": "No assignment ID"}), 400

    conn = get_db_connection()
    assignment = conn.execute("SELECT * FROM assignments WHERE id = ? AND user_id = ?", 
                             (assignment_id, user_id)).fetchone()
    if not assignment:
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403

    if completed:
        conn.execute("""
            UPDATE assignments SET completed = 1, time_spent_minutes = ? WHERE id = ?
        """, (time_spent_minutes, assignment_id))
        
        # Get user before update
        user = conn.execute("SELECT xp, level, streak FROM users WHERE id = ?", (user_id,)).fetchone()
        
        # Dynamic XP calculation based on time and consistency
        base_xp = max(10, min(100, time_spent_minutes))
        
        # Bonus for maintaining streak
        streak_bonus = min(50, user["streak"] * 5) if user else 0
        
        xp_earned = base_xp + streak_bonus
        new_xp = user["xp"] + xp_earned
        
        # Calculate new level
        new_level = calculate_level_from_xp(new_xp)
        
        # Update user with new XP, level, and streak
        conn.execute("""
            UPDATE users SET xp = ?, level = ?, streak = streak + 1 WHERE id = ?
        """, (new_xp, new_level, user_id))
        
    else:
        conn.execute("UPDATE assignments SET completed = 0 WHERE id = ?", (assignment_id,))
        xp_earned = 0

    conn.commit()
    conn.close()

    return jsonify({
        "success": True, 
        "xp_earned": xp_earned if completed else 0,
        "new_level": new_level if completed else 0
    })

def calculate_level_from_xp(xp):
    """Calculate level based on total XP (exponential scale)"""
    # Levels require increasing XP: Level 1 = 0 XP, Level 2 = 100 XP, Level 3 = 300 XP, etc.
    if xp < 100:
        return 1
    elif xp < 300:
        return 2
    elif xp < 600:
        return 3
    elif xp < 1000:
        return 4
    elif xp < 1500:
        return 5
    elif xp < 2100:
        return 6
    elif xp < 2800:
        return 7
    elif xp < 3600:
        return 8
    elif xp < 4500:
        return 9
    else:
        return 10 + (xp - 4500) // 1000

@app.route("/delete_assignment/<int:assignment_id>", methods=["POST"])
@login_required
def delete_assignment(assignment_id):
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    assignment = conn.execute("SELECT * FROM assignments WHERE id = ? AND user_id = ?", 
                             (assignment_id, user_id)).fetchone()
    if not assignment:
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403

    conn.execute("DELETE FROM assignments WHERE id = ?", (assignment_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ================= ENHANCED AI FEATURES =================

@app.route("/api/ai/analyze_workload", methods=["GET"])
@login_required
def analyze_workload():
    """Enhanced workload analysis with AI insights"""
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    assignments = conn.execute("""
        SELECT a.id, a.title, a.due_date, a.completed, a.time_spent_minutes, c.name as class_name
        FROM assignments a
        JOIN classes c ON a.class_id = c.id
        WHERE a.user_id = ? AND a.completed = 0
        ORDER BY a.due_date ASC
    """, (user_id,)).fetchall()
    
    # Get user patterns
    patterns = AdaptiveLearningEngine.analyze_user_patterns(user_id, conn)
    conn.close()
    
    today = date.today()
    workload = {
        "overdue": 0,
        "this_week": 0,
        "next_week": 0,
        "later": 0,
        "total_assignments": len(assignments),
        "assignments_by_class": {},
        "estimated_total_hours": 0,
        "stress_level": "low",
        "recommendations": []
    }
    
    total_minutes = 0
    for a in assignments:
        if a["due_date"]:
            due = datetime.strptime(a["due_date"], "%Y-%m-%d").date()
            days_until = (due - today).days
            
            if days_until < 0:
                workload["overdue"] += 1
            elif days_until <= 7:
                workload["this_week"] += 1
            elif days_until <= 14:
                workload["next_week"] += 1
            else:
                workload["later"] += 1
            
            class_name = a["class_name"]
            workload["assignments_by_class"][class_name] = \
                workload["assignments_by_class"].get(class_name, 0) + 1
            
            # Estimate time for incomplete assignments
            estimated_time = a["time_spent_minutes"] if a["time_spent_minutes"] > 0 else 60
            total_minutes += estimated_time
    
    workload["estimated_total_hours"] = round(total_minutes / 60, 1)
    
    # Determine stress level
    if workload["overdue"] > 2 or workload["this_week"] > 5:
        workload["stress_level"] = "high"
        workload["recommendations"].append("⚠️ High workload detected. Consider breaking tasks into smaller chunks.")
    elif workload["this_week"] > 3:
        workload["stress_level"] = "medium"
        workload["recommendations"].append("📊 Moderate workload. Stay consistent with your study schedule.")
    else:
        workload["stress_level"] = "low"
        workload["recommendations"].append("✅ Manageable workload. Great job staying on top of things!")
    
    if patterns:
        if patterns['recent_trend'] == 'declining':
            workload["recommendations"].append("📉 Your productivity has decreased recently. Try studying during your peak hours.")
        elif patterns['recent_trend'] == 'improving':
            workload["recommendations"].append("📈 Your productivity is improving! Keep up the momentum!")
    
    return jsonify(workload)

@app.route("/api/ai/predict_study_time", methods=["POST"])
@login_required
def predict_study_time():
    """ML-based study time prediction with Hugging Face fallback"""
    user_id = session["user"]["id"]
    data = request.get_json()
    assignment_title = data.get("title", "")
    class_name = data.get("class_name", "")
    
    conn = get_db_connection()
    
    # Get all historical data for ML prediction
    past_assignments = conn.execute("""
        SELECT a.title, a.time_spent_minutes, c.name as class_name
        FROM assignments a
        JOIN classes c ON a.class_id = c.id
        WHERE a.user_id = ? AND a.completed = 1 AND a.time_spent_minutes > 0
        ORDER BY a.id DESC
        LIMIT 50
    """, (user_id,)).fetchall()
    
    conn.close()
    
    # Use ML predictor
    history = [dict(a) for a in past_assignments]
    features = StudyTimePredictor.extract_features(assignment_title, class_name, history)
    predicted_time = StudyTimePredictor.predict(features)
    
    # Try to enhance with Hugging Face AI (optional)
    try:
        prompt = f"""Analyze this assignment and estimate study time in minutes.
Assignment: {assignment_title}
Class: {class_name}
Historical average: {features['user_avg_time']} minutes
Class average: {features['class_avg_time']} minutes

Respond with ONLY a number (minutes between 15-240)."""

        response = query_huggingface(
            HF_MODELS["text_generation"],
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 10,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }
        )
        
        if response and isinstance(response, list) and len(response) > 0:
            ai_prediction = response[0].get("generated_text", "").strip()
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', ai_prediction)
            if numbers:
                ai_time = int(numbers[0])
                if 15 <= ai_time <= 240:
                    # Average AI prediction with ML prediction
                    predicted_time = int((predicted_time + ai_time) / 2)
    except Exception as e:
        print(f"HF Enhancement failed: {e}")
    
    return jsonify({
        "predicted_minutes": predicted_time,
        "predicted_hours": round(predicted_time / 60, 1),
        "confidence": "high" if len(history) > 10 else "medium",
        "based_on_sessions": len(history)
    })

@app.route("/api/ai/optimal_study_times", methods=["GET"])
@login_required
def optimal_study_times():
    """AI-enhanced optimal study time recommendation"""
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    patterns = AdaptiveLearningEngine.analyze_user_patterns(user_id, conn)
    conn.close()
    
    if not patterns:
        return jsonify({
            "optimal_hours": [14, 16, 20],
            "recommendations": ["02:00 PM - 04:00 PM", "04:00 PM - 06:00 PM", "08:00 PM - 10:00 PM"],
            "total_sessions_analyzed": 0,
            "confidence": "low",
            "insights": ["Not enough data yet. Start tracking your study sessions!"]
        })
    
    optimal_hours = patterns['best_hours']
    insights = []
    
    if patterns['avg_session_length'] < 30:
        insights.append("💡 Your sessions are short. Try 45-60 minute focused blocks.")
    elif patterns['avg_session_length'] > 120:
        insights.append("⏰ Long sessions detected. Consider taking breaks every 90 minutes.")
    else:
        insights.append("✅ Your session length is optimal!")
    
    if patterns['recent_trend'] == 'improving':
        insights.append("🚀 Your study consistency is improving!")
    elif patterns['recent_trend'] == 'declining':
        insights.append("📉 Try to maintain regular study times for better results.")
    
    # Day of week insights
    productive_day = max(patterns['productive_days'].items(), key=lambda x: x[1])[0] if patterns['productive_days'] else 1
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    insights.append(f"📅 You're most productive on {day_names[productive_day]}s")
    
    return jsonify({
        "optimal_hours": optimal_hours,
        "recommendations": [f"{h:02d}:00 - {(h+2)%24:02d}:00" for h in optimal_hours],
        "total_sessions_analyzed": patterns['total_sessions'],
        "confidence": "high" if patterns['total_sessions'] > 20 else "medium",
        "insights": insights,
        "avg_session_minutes": int(patterns['avg_session_length'])
    })

@app.route("/generate_plan", methods=["POST"])
@login_required
def generate_plan():
    """Advanced AI-powered study schedule with Hugging Face"""
    user_id = session["user"]["id"]
    conn = get_db_connection()

    try:
        # Fetch assignments
        assignments = conn.execute("""
            SELECT a.id, a.title, a.due_date, a.time_spent_minutes,
                   c.name as class_name
            FROM assignments a
            JOIN classes c ON a.class_id = c.id
            WHERE c.user_id = ? AND a.completed = 0
            ORDER BY a.due_date ASC
        """, (user_id,)).fetchall()

        if not assignments:
            return jsonify({"success": False,
                            "message": "No assignments to schedule"})

        # Analyze study patterns
        patterns = AdaptiveLearningEngine.analyze_user_patterns(user_id, conn)
        if not patterns:
            patterns = {
                'best_hours': [14, 16, 20],
                'avg_session_length': 60
            }

        # Fetch history
        history = conn.execute("""
            SELECT a.title, a.time_spent_minutes, c.name as class_name
            FROM assignments a
            JOIN classes c ON a.class_id = c.id
            WHERE a.user_id = ? AND a.completed = 1
        """, (user_id,)).fetchall()

        history_list = [dict(h) for h in history]

        # Prepare predicted study times
        assignment_data = []
        for a in assignments:
            features = StudyTimePredictor.extract_features(
                a["title"], a["class_name"], history_list
            )
            predicted_time = StudyTimePredictor.predict(features)

            assignment_data.append({
                "id": a["id"],
                "title": a["title"],
                "class": a["class_name"],
                "due_date": a["due_date"] or "No deadline",
                "predicted_time": predicted_time
            })

        # Generate smart schedule
        schedule = []
        current_date = datetime.now().date()

        sorted_assignments = sorted(
            assignment_data,
            key=lambda x: (
                datetime.strptime(x["due_date"], "%Y-%m-%d").date()
                if x["due_date"] != "No deadline"
                else current_date + timedelta(days=30)
            )
        )

        study_hours = patterns['best_hours']
        day_offset = 0
        hour_index = 0

        for assignment in sorted_assignments:
            if assignment["due_date"] != "No deadline":
                due_date = datetime.strptime(
                    assignment["due_date"], "%Y-%m-%d"
                ).date()
                days_until_due = (due_date - current_date).days

                if day_offset >= days_until_due - 1:
                    day_offset = max(0, days_until_due - 2)

            study_date = current_date + timedelta(days=day_offset)

            time_needed = assignment["predicted_time"]
            max_session = max(
                30, int(patterns['avg_session_length'])
            )

            while time_needed > 0:
                session_duration = min(time_needed, max_session)
                study_hour = study_hours[hour_index % len(study_hours)]

                schedule.append({
                    "assignment_id": assignment["id"],
                    "title": assignment["title"],
                    "class": assignment["class"],
                    "date": study_date.strftime("%Y-%m-%d"),
                    "start_time": f"{study_hour:02d}:00",
                    "duration_minutes": session_duration
                })

                time_needed -= session_duration
                hour_index += 1

                if hour_index % len(study_hours) == 0:
                    day_offset += 1
                    study_date = current_date + timedelta(days=day_offset)

        # Optional: AI advice
        ai_advice = None
        try:
            prompt = (
                "Create short smart scheduling advice:\n"
                f"{json.dumps(assignment_data[:5], indent=2)}\n\n"
                f"Best study hours: {study_hours}\n"
                f"Average session: {int(patterns['avg_session_length'])} min\n"
                "Give 3 tips, each one line."
            )

            response = query_huggingface(
                HF_MODELS["text_generation"],
                {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 120,
                        "temperature": 0.7
                    }
                }
            )

            if response and isinstance(response, list):
                ai_advice = response[0].get("generated_text", "").strip()

        except Exception:
            ai_advice = None

        # Reset study plan
        conn.execute("DELETE FROM study_plan WHERE user_id = ?", (user_id,))

        for block in schedule:
            start_dt = datetime.strptime(
                f"{block['date']} {block['start_time']}", "%Y-%m-%d %H:%M"
            )
            end_dt = start_dt + timedelta(
                minutes=block['duration_minutes']
            )

            conn.execute("""
                INSERT INTO study_plan (user_id, title, start, end)
                VALUES (?, ?, ?, ?)
            """, (
                user_id,
                f"{block['class']}: {block['title']}",
                start_dt.isoformat(),
                end_dt.isoformat()
            ))

        conn.commit()

        return jsonify({
            "success": True,
            "message": f"Generated {len(schedule)} study blocks",
            "schedule": schedule,
            "total_study_hours": sum(b["duration_minutes"] for b in schedule) / 60,
            "assignments_covered": len(sorted_assignments),
            "ai_advice": ai_advice
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        conn.close()

@app.route("/study_planner")
@login_required
def study_planner():
    user_id = session["user"]["id"]
    conn = get_db()
    
    events = conn.execute("""
        SELECT id, title, start, end FROM study_plan
        WHERE user_id = ?
        ORDER BY start ASC
    """, (user_id,)).fetchall()

    study_events = [
        {
            "id": e["id"],
            "title": e["title"],
            "start": e["start"],
            "end": e["end"]
        } for e in events
    ]

    return render_template("study_planner.html", title="Study Planner", study_events=study_events)

@app.route("/planner")
@login_required
def planner():
    return redirect(url_for("study_planner"))

@app.route("/delete_study_block/<int:block_id>", methods=["DELETE"])
@login_required
def delete_study_block(block_id):
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    block = conn.execute("SELECT * FROM study_plan WHERE id = ? AND user_id = ?", 
                        (block_id, user_id)).fetchone()
    if not block:
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403
    
    conn.execute("DELETE FROM study_plan WHERE id = ?", (block_id,))
    conn.commit()
    conn.close()
    return ("", 204)

@app.route("/update_study_block/<int:block_id>", methods=["POST"])
@login_required
def update_study_block(block_id):
    user_id = session["user"]["id"]
    data = request.get_json()
    
    start = data.get("start")
    end = data.get("end")
    
    if not start or not end:
        return jsonify({"error": "Start and end times required"}), 400
    
    conn = get_db_connection()
    block = conn.execute("SELECT * FROM study_plan WHERE id = ? AND user_id = ?", 
                        (block_id, user_id)).fetchone()
    if not block:
        conn.close()
        return jsonify({"error": "Unauthorized"}), 403
    
    conn.execute("""
        UPDATE study_plan SET start = ?, end = ? WHERE id = ?
    """, (start, end, block_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ================= ADVANCED AI INSIGHTS =================

@app.route("/api/ai/get_reminders", methods=["GET"])
@login_required
def get_reminders():
    """Smart AI-powered reminders with priority scoring"""
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    today = date.today()
    upcoming = conn.execute("""
        SELECT a.id, a.title, a.due_date, c.name as class_name
        FROM assignments a
        JOIN classes c ON a.class_id = c.id
        WHERE a.user_id = ? AND a.completed = 0 AND a.due_date IS NOT NULL
        ORDER BY a.due_date ASC
        LIMIT 15
    """, (user_id,)).fetchall()
    
    conn.close()
    
    reminders = []
    for a in upcoming:
        due = datetime.strptime(a["due_date"], "%Y-%m-%d").date()
        days_left = (due - today).days
        
        # Calculate priority score
        if days_left < 0:
            urgency = "critical"
            priority = 100
        elif days_left == 0:
            urgency = "critical"
            priority = 95
        elif days_left == 1:
            urgency = "high"
            priority = 85
        elif days_left <= 3:
            urgency = "high"
            priority = 70
        elif days_left <= 7:
            urgency = "medium"
            priority = 50
        else:
            urgency = "low"
            priority = 30
        
        # Generate contextual messages
        if days_left < 0:
            message = f"🚨 CRITICAL: {a['title']} ({a['class_name']}) was due {abs(days_left)} days ago!"
            action = "Complete immediately"
        elif days_left == 0:
            message = f"🔥 DUE TODAY: {a['title']} ({a['class_name']})"
            action = "Prioritize now"
        elif days_left == 1:
            message = f"⚡ Due tomorrow: {a['title']} ({a['class_name']})"
            action = "Work on today"
        elif days_left <= 3:
            message = f"⏰ {a['title']} ({a['class_name']}) due in {days_left} days"
            action = "Start soon"
        elif days_left <= 7:
            message = f"📅 {a['title']} ({a['class_name']}) due in {days_left} days"
            action = "Plan ahead"
        else:
            message = f"📆 {a['title']} ({a['class_name']}) due in {days_left} days"
            action = "Keep on radar"
        
        reminders.append({
            "id": a["id"],
            "message": message,
            "urgency": urgency,
            "priority": priority,
            "days_left": days_left,
            "due_date": a["due_date"],
            "action": action
        })
    
    # Sort by priority
    reminders.sort(key=lambda x: x["priority"], reverse=True)
    
    return jsonify({
        "reminders": reminders,
        "critical_count": sum(1 for r in reminders if r["urgency"] == "critical"),
        "high_count": sum(1 for r in reminders if r["urgency"] == "high")
    })

@app.route("/api/ai/productivity_report", methods=["GET"])
@login_required
def productivity_report():
    """Generate comprehensive productivity analytics"""
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    # Get completion stats
    total_completed = conn.execute("""
        SELECT COUNT(*) as count, SUM(time_spent_minutes) as total_time
        FROM assignments
        WHERE user_id = ? AND completed = 1
    """, (user_id,)).fetchone()
    
    total_pending = conn.execute("""
        SELECT COUNT(*) as count
        FROM assignments
        WHERE user_id = ? AND completed = 0
    """, (user_id,)).fetchone()
    
    # Get recent activity (last 7 days)
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    recent_completed = conn.execute("""
        SELECT COUNT(*) as count
        FROM assignments
        WHERE user_id = ? AND completed = 1 AND started_at >= ?
    """, (user_id, week_ago)).fetchone()
    
    # Get patterns
    patterns = AdaptiveLearningEngine.analyze_user_patterns(user_id, conn)
    
    # Get class performance
    class_stats = conn.execute("""
        SELECT c.name, 
               COUNT(*) as total,
               SUM(CASE WHEN a.completed = 1 THEN 1 ELSE 0 END) as completed
        FROM assignments a
        JOIN classes c ON a.class_id = c.id
        WHERE a.user_id = ?
        GROUP BY c.name
    """, (user_id,)).fetchall()
    
    conn.close()
    
    # Calculate metrics
    completion_rate = 0
    if total_completed["count"] + total_pending["count"] > 0:
        completion_rate = round(
            (total_completed["count"] / (total_completed["count"] + total_pending["count"])) * 100, 
            1
        )
    
    total_hours = round((total_completed["total_time"] or 0) / 60, 1)
    
    # Generate insights
    insights = []
    
    if completion_rate >= 80:
        insights.append("🌟 Excellent completion rate! You're staying on top of your work.")
    elif completion_rate >= 60:
        insights.append("👍 Good completion rate. Keep up the consistency!")
    else:
        insights.append("💪 There's room for improvement. Try breaking tasks into smaller chunks.")
    
    if recent_completed["count"] >= 5:
        insights.append(f"🔥 You've completed {recent_completed['count']} assignments this week!")
    elif recent_completed["count"] > 0:
        insights.append(f"📈 {recent_completed['count']} assignments completed this week. Keep building momentum!")
    else:
        insights.append("⏰ No assignments completed this week. Time to catch up!")
    
    if patterns and patterns['recent_trend'] == 'improving':
        insights.append("📊 Your productivity trend is improving!")
    elif patterns and patterns['recent_trend'] == 'declining':
        insights.append("📉 Your productivity has dipped. Try scheduling more consistent study blocks.")
    
    class_performance = []
    for stat in class_stats:
        rate = round((stat["completed"] / stat["total"]) * 100, 1) if stat["total"] > 0 else 0
        class_performance.append({
            "class": stat["name"],
            "completion_rate": rate,
            "total": stat["total"],
            "completed": stat["completed"]
        })
    
    return jsonify({
        "completion_rate": completion_rate,
        "total_completed": total_completed["count"],
        "total_pending": total_pending["count"],
        "total_study_hours": total_hours,
        "weekly_completed": recent_completed["count"],
        "insights": insights,
        "class_performance": class_performance,
        "avg_session_minutes": int(patterns['avg_session_length']) if patterns else 60,
        "best_study_hours": patterns['best_hours'] if patterns else [14, 16, 20]
    })

@app.route("/api/ai/smart_break_reminder", methods=["GET"])
@login_required
def smart_break_reminder():
    """Remind users to take breaks based on study duration"""
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    # Check if user has been studying too long
    recent_session = conn.execute("""
        SELECT started_at, time_spent_minutes
        FROM assignments
        WHERE user_id = ? AND completed = 1
        ORDER BY started_at DESC
        LIMIT 1
    """, (user_id,)).fetchone()
    
    conn.close()
    
    if not recent_session or not recent_session["started_at"]:
        return jsonify({
            "should_break": False,
            "message": "Start a study session to track your breaks!"
        })
    
    # Simple break logic
    study_minutes = recent_session["time_spent_minutes"] or 0
    
    if study_minutes >= 90:
        return jsonify({
            "should_break": True,
            "message": "🧘 You've been studying for 90+ minutes. Take a 15-minute break!",
            "break_type": "long",
            "suggested_duration": 15
        })
    elif study_minutes >= 50:
        return jsonify({
            "should_break": True,
            "message": "☕ Good progress! Consider a 5-10 minute break.",
            "break_type": "short",
            "suggested_duration": 7
        })
    else:
        return jsonify({
            "should_break": False,
            "message": "💪 Keep up the focus! Break recommended after 50 minutes.",
            "minutes_until_break": 50 - study_minutes
        })

# ================= GAMIFICATION ENHANCEMENTS =================

@app.route("/api/gamification/leaderboard", methods=["GET"])
@login_required
def leaderboard():
    """Get leaderboard with privacy protection"""
    conn = get_db_connection()
    
    top_users = conn.execute("""
        SELECT name, xp, level, streak
        FROM users
        ORDER BY xp DESC
        LIMIT 10
    """).fetchall()
    
    conn.close()
    
    leaderboard_data = []
    for idx, user in enumerate(top_users, 1):
        # Anonymize names except for current user
        display_name = user["name"] if user["name"] == session["user"]["name"] else f"Student {idx}"
        
        leaderboard_data.append({
            "rank": idx,
            "name": display_name,
            "xp": user["xp"],
            "level": user["level"],
            "streak": user["streak"]
        })
    
    return jsonify({"leaderboard": leaderboard_data})

@app.route("/api/gamification/achievements", methods=["GET"])
@login_required
def get_achievements():
    """Get user achievements"""
    user_id = session["user"]["id"]
    conn = get_db_connection()
    
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    completed = conn.execute("""
        SELECT COUNT(*) as count FROM assignments WHERE user_id = ? AND completed = 1
    """, (user_id,)).fetchone()
    
    conn.close()
    
    achievements = []
    
    # Streak achievements
    if user["streak"] >= 30:
        achievements.append({"name": "🔥 30-Day Streak Master", "unlocked": True})
    elif user["streak"] >= 14:
        achievements.append({"name": "⚡ 2-Week Warrior", "unlocked": True})
    elif user["streak"] >= 7:
        achievements.append({"name": "📅 Week Champion", "unlocked": True})
    
    # Completion achievements
    if completed["count"] >= 100:
        achievements.append({"name": "💯 Century Club", "unlocked": True})
    elif completed["count"] >= 50:
        achievements.append({"name": "⭐ Half-Century Hero", "unlocked": True})
    elif completed["count"] >= 10:
        achievements.append({"name": "🎯 First Ten", "unlocked": True})
    
    # XP achievements
    if user["xp"] >= 5000:
        achievements.append({"name": "👑 Grand Master", "unlocked": True})
    elif user["xp"] >= 1000:
        achievements.append({"name": "🏆 Expert Scholar", "unlocked": True})
    elif user["xp"] >= 100:
        achievements.append({"name": "🌟 Rising Star", "unlocked": True})
    
    return jsonify({
        "achievements": achievements,
        "total_unlocked": len(achievements)
    })

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)