from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import pickle, numpy as np, os

app = Flask(__name__)
app.secret_key = 'smarthealth_secret_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ─── Load ML models ───────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
model    = pickle.load(open(os.path.join(BASE,'models','best_model.pkl'),'rb'))
encoder  = pickle.load(open(os.path.join(BASE,'models','disease_encoder.pkl'),'rb'))
symptoms = pickle.load(open(os.path.join(BASE,'models','symptoms.pkl'),'rb'))

DISEASE_INFO = {
    'Flu':              {'specialist':'General Physician','precautions':'Rest, fluids, avoid crowds','severity':'Moderate'},
    'Common Cold':      {'specialist':'General Physician','precautions':'Stay warm, drink fluids, rest','severity':'Mild'},
    'Malaria':          {'specialist':'Infectious Disease','precautions':'Mosquito nets, antimalarial meds','severity':'High'},
    'Dengue':           {'specialist':'Infectious Disease','precautions':'Hydration, avoid NSAIDs, rest','severity':'High'},
    'Typhoid':          {'specialist':'Gastroenterologist','precautions':'Clean water, hygiene, antibiotics','severity':'High'},
    'Diabetes':         {'specialist':'Endocrinologist','precautions':'Diet control, exercise, monitor sugar','severity':'Chronic'},
    'Hypertension':     {'specialist':'Cardiologist','precautions':'Low-salt diet, exercise, stress management','severity':'Chronic'},
    'Pneumonia':        {'specialist':'Pulmonologist','precautions':'Antibiotics, rest, hydration','severity':'High'},
    'Asthma':           {'specialist':'Pulmonologist','precautions':'Avoid triggers, carry inhaler','severity':'Moderate'},
    'COVID-19':         {'specialist':'General Physician','precautions':'Isolation, hydration, oxygen monitoring','severity':'High'},
    'Migraine':         {'specialist':'Neurologist','precautions':'Avoid triggers, dark room, pain relief','severity':'Moderate'},
    'Gastritis':        {'specialist':'Gastroenterologist','precautions':'Bland diet, antacids, avoid spicy food','severity':'Moderate'},
    'Anemia':           {'specialist':'Hematologist','precautions':'Iron-rich diet, supplements','severity':'Moderate'},
    'Arthritis':        {'specialist':'Rheumatologist','precautions':'Physical therapy, anti-inflammatory meds','severity':'Chronic'},
    'Jaundice':         {'specialist':'Hepatologist','precautions':'Rest, hydration, avoid alcohol','severity':'High'},
}

# ─── Database Models ───────────────────────────────────────────────────────────
class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    age        = db.Column(db.Integer)
    gender     = db.Column(db.String(10))
    phone      = db.Column(db.String(15))
    blood_group= db.Column(db.String(5))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'))
    disease     = db.Column(db.String(100))
    confidence  = db.Column(db.Float)
    symptoms    = db.Column(db.String(500))
    severity    = db.Column(db.String(20))
    specialist  = db.Column(db.String(100))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'))
    doctor_name = db.Column(db.String(100))
    department  = db.Column(db.String(100))
    date        = db.Column(db.String(20))
    time        = db.Column(db.String(20))
    reason      = db.Column(db.String(300))
    status      = db.Column(db.String(20), default='Scheduled')
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

# ─── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name  = request.form['name']
        email = request.form['email']
        pwd   = request.form['password']
        age   = request.form.get('age',0)
        gender= request.form.get('gender','')
        phone = request.form.get('phone','')
        bg    = request.form.get('blood_group','')
        if User.query.filter_by(email=email).first():
            flash('Email already registered!','danger')
            return redirect(url_for('register'))
        user = User(name=name, email=email,
                    password=generate_password_hash(pwd),
                    age=age, gender=gender, phone=phone, blood_group=bg)
        db.session.add(user); db.session.commit()
        flash('Registration successful! Please login.','success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        pwd   = request.form['password']
        user  = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, pwd):
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash(f'Welcome back, {user.name}!','success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials!','danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.','info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    preds = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).limit(5).all()
    apts  = Appointment.query.filter_by(user_id=user.id).order_by(Appointment.created_at.desc()).limit(5).all()
    total_preds = Prediction.query.filter_by(user_id=user.id).count()
    total_apts  = Appointment.query.filter_by(user_id=user.id).count()
    return render_template('dashboard.html', user=user, predictions=preds,
                           appointments=apts, total_preds=total_preds, total_apts=total_apts)

@app.route('/prediction', methods=['GET','POST'])
@login_required
def prediction():
    result = None
    if request.method == 'POST':
        selected = request.form.getlist('symptoms')
        vec = [1 if s in selected else 0 for s in symptoms]
        probs = model.predict_proba([vec])[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top_disease = encoder.classes_[top_idx[0]]
        confidence  = round(probs[top_idx[0]] * 100, 1)
        info = DISEASE_INFO.get(top_disease, {'specialist':'General Physician','precautions':'Consult a doctor','severity':'Unknown'})
        # Top 3 predictions
        top3 = [(encoder.classes_[i], round(probs[i]*100,1)) for i in top_idx]
        pred = Prediction(user_id=session['user_id'], disease=top_disease,
                          confidence=confidence, symptoms=', '.join(selected),
                          severity=info['severity'], specialist=info['specialist'])
        db.session.add(pred); db.session.commit()
        result = {'disease': top_disease, 'confidence': confidence,
                  'top3': top3, **info, 'symptoms_selected': selected}
    return render_template('prediction.html', symptoms=symptoms, result=result)

@app.route('/appointment', methods=['GET','POST'])
@login_required
def appointment():
    if request.method == 'POST':
        apt = Appointment(
            user_id    = session['user_id'],
            doctor_name= request.form['doctor_name'],
            department = request.form['department'],
            date       = request.form['date'],
            time       = request.form['time'],
            reason     = request.form['reason']
        )
        db.session.add(apt); db.session.commit()
        flash('Appointment booked successfully!','success')
        return redirect(url_for('appointment'))
    apts = Appointment.query.filter_by(user_id=session['user_id']).order_by(Appointment.created_at.desc()).all()
    return render_template('appointment.html', appointments=apts)

@app.route('/appointment/cancel/<int:apt_id>')
@login_required
def cancel_appointment(apt_id):
    apt = Appointment.query.get_or_404(apt_id)
    if apt.user_id == session['user_id']:
        apt.status = 'Cancelled'
        db.session.commit()
        flash('Appointment cancelled.','info')
    return redirect(url_for('appointment'))

@app.route('/reports')
@login_required
def reports():
    user  = User.query.get(session['user_id'])
    preds = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).all()
    apts  = Appointment.query.filter_by(user_id=user.id).order_by(Appointment.created_at.desc()).all()
    # Stats for charts
    disease_counts = {}
    for p in preds:
        disease_counts[p.disease] = disease_counts.get(p.disease, 0) + 1
    return render_template('reports.html', user=user, predictions=preds,
                           appointments=apts, disease_counts=disease_counts)

@app.route('/api/stats')
@login_required
def api_stats():
    uid = session['user_id']
    preds = Prediction.query.filter_by(user_id=uid).all()
    counts = {}
    for p in preds:
        counts[p.disease] = counts.get(p.disease,0)+1
    return jsonify({'disease_counts': counts, 'total': len(preds)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ Database initialized!")
    app.run(debug=True, port=5000)
