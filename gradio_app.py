import os
import pickle
import warnings
import numpy as np
import gradio as gr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load model — works whether model.pkl sits next to this script
# (D:\AQI_prediction_Model\model.pkl) or inside a models\ subfolder
# (D:\AQI_prediction_Model\models\model.pkl)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_PATHS = [
    os.path.join(SCRIPT_DIR, "model.pkl"),
    os.path.join(SCRIPT_DIR, "models", "model.pkl"),
]

MODEL_PATH = next((p for p in CANDIDATE_PATHS if os.path.isfile(p)), None)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find model.pkl. Checked:\n  " + "\n  ".join(CANDIDATE_PATHS) +
        "\nPlace model.pkl next to this script, or inside a 'models' subfolder."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Order matters — must match training order
FEATURES = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "AH",
]

# label, min, max, default, step, unit
FIELD_SPECS = {
    "CO(GT)":        ("CO (ground truth)",            0.0, 12.0, 2.0,   0.1, "mg/m³"),
    "PT08.S1(CO)":   ("PT08.S1 — CO sensor",           700, 2100, 1100,  1,   "resp."),
    "NMHC(GT)":      ("NMHC (ground truth)",           0,   1200, 150,   1,   "µg/m³"),
    "C6H6(GT)":      ("Benzene C6H6 (ground truth)",   0.0, 60.0, 10.0,  0.1, "µg/m³"),
    "PT08.S2(NMHC)": ("PT08.S2 — NMHC sensor",         300, 2200, 900,   1,   "resp."),
    "NOx(GT)":       ("NOx (ground truth)",            0,   1500, 200,   1,   "ppb"),
    "PT08.S3(NOx)":  ("PT08.S3 — NOx sensor",          300, 2700, 900,   1,   "resp."),
    "NO2(GT)":       ("NO2 (ground truth)",            0,   350,  100,   1,   "µg/m³"),
    "PT08.S4(NO2)":  ("PT08.S4 — NO2 sensor",          700,  2700, 1500,  1,   "resp."),
    "PT08.S5(O3)":   ("PT08.S5 — O3 sensor",           300,  2500, 1000,  1,   "resp."),
    "T":             ("Temperature",                   -5.0, 45.0, 18.0,  0.1, "°C"),
    "AH":            ("Absolute Humidity",              0.0, 2.5,  0.9,   0.01,"g/m³"),
}

GROUPS = [
    ("🧪 Ground-Truth Gas Concentrations", ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]),
    ("📡 Sensor Responses (PT08 array)", ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]),
    ("🌡️ Weather Conditions", ["T", "AH"]),
]

EXAMPLES = [
    [2.0, 1100, 150, 10.0, 900, 200, 900, 100, 1500, 1000, 18.0, 0.90],
    [0.6, 1010, 40,  4.5,  750, 80,  1200,60,  1150, 850,  25.5, 1.20],
    [4.2, 1350, 300, 18.0, 1150,450, 650, 180, 1900, 1350, 8.0,  0.55],
]

# ---------------------------------------------------------------------------
# Styling — animated background, glass cards, and 4 switchable colour themes
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root, body[data-theme="ocean"] {
    --grad-a: #0f766e; --grad-b: #0891b2; --grad-c: #06b6d4;
    --accent: #0ea5e9; --accent-soft: #cffafe;
    --bg-a: #ecfeff; --bg-b: #f0fdfa;
}
body[data-theme="sunset"] {
    --grad-a: #ea580c; --grad-b: #db2777; --grad-c: #f59e0b;
    --accent: #f97316; --accent-soft: #ffedd5;
    --bg-a: #fff7ed; --bg-b: #fef2f2;
}
body[data-theme="forest"] {
    --grad-a: #166534; --grad-b: #65a30d; --grad-c: #16a34a;
    --accent: #22c55e; --accent-soft: #dcfce7;
    --bg-a: #f0fdf4; --bg-b: #ecfccb;
}
body[data-theme="midnight"] {
    --grad-a: #4338ca; --grad-b: #7c3aed; --grad-c: #c026d3;
    --accent: #a855f7; --accent-soft: #ede9fe;
    --bg-a: #f5f3ff; --bg-b: #faf5ff;
}

* { font-family: 'Poppins', sans-serif; }
h1, h2, h3, .gauge-value { font-family: 'Space Grotesk', sans-serif; }

.gradio-container {
    max-width: 1120px !important;
    margin: auto !important;
    position: relative;
    z-index: 1;
}

/* animated floating blobs behind everything */
body::before, body::after {
    content: "";
    position: fixed;
    width: 480px; height: 480px;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.35;
    z-index: 0;
    animation: float 16s ease-in-out infinite;
    background: linear-gradient(135deg, var(--grad-b), var(--grad-c));
}
body::before { top: -150px; left: -150px; }
body::after { bottom: -150px; right: -150px; animation-delay: -8s; background: linear-gradient(135deg, var(--grad-a), var(--grad-b)); }

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(40px, 60px) scale(1.15); }
}

#app-header {
    background: linear-gradient(120deg, var(--grad-a), var(--grad-b) 55%, var(--grad-c));
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    border-radius: 20px;
    padding: 30px 34px;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
#app-header h1 { margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 700; }
#app-header p { margin: 0; opacity: 0.94; font-size: 0.98rem; }

.theme-row { margin-bottom: 14px; }

.group-card {
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 16px;
    padding: 14px 18px 6px 18px;
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(10px);
    margin-bottom: 12px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.group-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.09);
}
.group-card .prose { font-weight: 600 !important; color: var(--grad-a); }

#predict-btn {
    background: linear-gradient(120deg, var(--grad-a), var(--grad-c)) !important;
    background-size: 200% 200% !important;
    animation: gradientShift 6s ease infinite;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.08rem !important;
    border-radius: 14px !important;
    height: 50px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    transition: transform 0.15s ease;
}
#predict-btn:hover { transform: translateY(-2px) scale(1.01); }

#result-panel {
    border-radius: 20px;
    padding: 20px;
    background: linear-gradient(160deg, var(--bg-a), var(--bg-b));
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 8px 26px rgba(0,0,0,0.07);
    text-align: center;
}
.gauge-value { font-size: 2.8rem; font-weight: 700; margin: 6px 0 0 0; }
.gauge-tag { font-size: 1rem; font-weight: 600; letter-spacing: 0.02em; }
.gauge-sub { color: #64748b; font-size: 0.85rem; margin-top: 2px; }

footer { display: none !important; }
"""

THEME_JS = """
(theme) => {
    document.body.setAttribute('data-theme', theme);
    return theme;
}
"""

INIT_JS = """
() => {
    document.body.setAttribute('data-theme', 'ocean');
}
"""

THEME_COLORS = {
    "🌊 Ocean":    {"a": "#0f766e", "b": "#0891b2", "c": "#06b6d4"},
    "🌅 Sunset":   {"a": "#ea580c", "b": "#db2777", "c": "#f59e0b"},
    "🌲 Forest":   {"a": "#166534", "b": "#65a30d", "c": "#16a34a"},
    "🌌 Midnight": {"a": "#4338ca", "b": "#7c3aed", "c": "#c026d3"},
}


def gauge_svg(pred: float, accent: str) -> str:
    """Semi-circular speedometer gauge with a needle pointing at `pred` (0-100)."""
    pred = max(0.0, min(100.0, pred))
    angle = -90 + (pred / 100.0) * 180  # -90 (0%) to +90 (100%)
    rad = np.radians(angle)
    cx, cy, r = 110, 110, 90
    nx = cx + r * 0.72 * np.sin(rad)
    ny = cy - r * 0.72 * np.cos(rad)

    return f"""
    <svg viewBox="0 0 220 130" width="240" height="145">
        <path d="M 20 110 A 90 90 0 0 1 68 30" fill="none" stroke="#ea580c" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
        <path d="M 68 30 A 90 90 0 0 1 152 30" fill="none" stroke="{accent}" stroke-width="14" stroke-linecap="round" opacity="0.9"/>
        <path d="M 152 30 A 90 90 0 0 1 200 110" fill="none" stroke="#2563eb" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
        <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#1e293b" stroke-width="4" stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="8" fill="#1e293b"/>
    </svg>
    """


def predict_rh(theme, *values):
    x = np.array(values, dtype=float).reshape(1, -1)
    pred = model.predict(x)[0]
    pred = float(np.clip(pred, 0, 100))

    if pred < 30:
        tag, color = "Dry", "#ea580c"
    elif pred < 60:
        tag, color = "Comfortable", "#0f766e"
    else:
        tag, color = "Humid", "#2563eb"

    accent = THEME_COLORS.get(theme, THEME_COLORS["🌊 Ocean"])["b"]
    gauge = gauge_svg(pred, accent)

    html = f"""
    <div id="result-panel">
        <div>{gauge}</div>
        <div class="gauge-value" style="color:{color}">{pred:.2f}%</div>
        <div class="gauge-tag" style="color:{color}">{tag}</div>
        <div class="gauge-sub">Predicted Relative Humidity</div>
    </div>
    """
    return html


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"), js=INIT_JS) as demo:
    gr.HTML(
        """
        <div id="app-header">
            <h1>💧 Relative Humidity Predictor</h1>
            <p>Air-quality sensor readings in → predicted RH (%) out, powered by a Random Forest model
            trained on the UCI Air Quality dataset.</p>
        </div>
        """
    )

    with gr.Row(elem_classes="theme-row"):
        theme_picker = gr.Radio(
            choices=list(THEME_COLORS.keys()),
            value="🌊 Ocean",
            label="🎨 Pick a theme",
        )

    inputs = []
    with gr.Row():
        with gr.Column(scale=1):
            for title, keys in GROUPS:
                with gr.Group(elem_classes="group-card"):
                    gr.Markdown(f"**{title}**")
                    for k in keys:
                        label, mn, mx, default, step, unit = FIELD_SPECS[k]
                        comp = gr.Slider(
                            minimum=mn, maximum=mx, value=default, step=step,
                            label=f"{label}  ({unit})",
                        )
                        inputs.append(comp)

        with gr.Column(scale=1):
            gr.Markdown("### Prediction")
            result = gr.HTML(
                """<div id="result-panel">
                       <div class="gauge-value" style="color:#94a3b8">—</div>
                       <div class="gauge-sub">Set the inputs and click Predict</div>
                   </div>"""
            )
            predict_btn = gr.Button("🔮 Predict RH", elem_id="predict-btn")

            gr.Markdown("#### Quick Examples")
            gr.Examples(
                examples=EXAMPLES,
                inputs=inputs,
                label="Click a row to load sample sensor data",
            )

    theme_picker.change(fn=None, inputs=theme_picker, outputs=None, js=THEME_JS)
    predict_btn.click(fn=predict_rh, inputs=[theme_picker] + inputs, outputs=result)

if __name__ == "__main__":
    print(f"Loaded model from: {MODEL_PATH}")
    demo.launch()