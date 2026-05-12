import streamlit as st
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import tempfile


import gdown
import os


# --- CONFIGURATION ---
selected_classes = ["PushUps", "BenchPress", "Diving", "JumpRope", "JumpingJack", "HulaHoop",
                    "JavelinThrow", "TennisSwing", "Basketball", "GolfSwing", "PlayingGuitar", 
                    "Drumming", "Rowing", "WalkingWithDog", "HighJump"]

CLASS_ICONS = {
    "PushUps": "💪", "BenchPress": "🏋️", "Diving": "🤿", "JumpRope": "🪢",
    "JumpingJack": "🤸", "HulaHoop": "⭕", "JavelinThrow": "🏹", "TennisSwing": "🎾",
    "Basketball": "🏀", "GolfSwing": "⛳", "PlayingGuitar": "🎸", "Drumming": "🥁",
    "Rowing": "🚣", "WalkingWithDog": "🐕", "HighJump": "🏃"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "har.pth"

# ─────────────────────────────────────────────
# BACKEND (100% UNCHANGED)
# ─────────────────────────────────────────────
class HARModel(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        backbone = models.efficientnet_v2_s(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.rgb_proj = nn.Linear(1280, 512)
        self.flow_proj = nn.Linear(128, 128)
        self.fusion_dim = 640
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim, nhead=8, dim_feedforward=1024,
            dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(0.5),
            nn.Linear(self.fusion_dim, num_classes)
        )

    def forward(self, rgb_seq, flow_seq):
        B, T, C, H, W = rgb_seq.shape
        rgb_reshaped = rgb_seq.view(B * T, C, H, W)
        rgb_feats = self.feature_extractor(rgb_reshaped)
        rgb_feats = torch.flatten(rgb_feats, 1)
        rgb_feats = self.rgb_proj(rgb_feats).view(B, T, -1)
        flow_feats = self.flow_proj(flow_seq)
        fused = torch.cat([rgb_feats, flow_feats], dim=-1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        fused = torch.cat((cls_tokens, fused), dim=1)
        out = self.transformer(fused)
        return self.classifier(out[:, 0, :])

def compute_farneback_flow(prev_gray, curr_gray):
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def extract_flow_features(flow, grid_size=4, num_bins=8):
    h, w = flow.shape[:2]
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    cell_h, cell_w = h // grid_size, w // grid_size
    features = []
    for row in range(grid_size):
        for col in range(grid_size):
            r0, r1 = row * cell_h, (row + 1) * cell_h
            c0, c1 = col * cell_w, (col + 1) * cell_w
            hist, _ = np.histogram(angle[r0:r1, c0:c1], bins=num_bins, range=(0, 360),
                                   weights=magnitude[r0:r1, c0:c1])
            norm = np.linalg.norm(hist) + 1e-6
            features.append(hist / norm)
    return np.concatenate(features).astype(np.float32)

def process_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None, None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    rgb_seq, flow_seq = [], []
    for t, frame in enumerate(frames):
        rgb_seq.append(val_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if t == 0:
            flow_feat = np.zeros(128, dtype=np.float32)
        else:
            flow = compute_farneback_flow(gray_frames[t - 1], gray_frames[t])
            flow_feat = extract_flow_features(flow)
        flow_seq.append(torch.tensor(flow_feat))
    return torch.stack(rgb_seq).unsqueeze(0), torch.stack(flow_seq).unsqueeze(0)

# @st.cache_resource
# def load_trained_model():
#     model = HARModel(num_classes=15).to(device)
#     if not os.path.exists(MODEL_PATH):
#         st.error(f"Model file `{MODEL_PATH}` not found. Place it in the same directory.")
#         return None
#     checkpoint = torch.load(MODEL_PATH, map_location=device)
#     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         state_dict = checkpoint["model_state_dict"]
#     else:
#         state_dict = checkpoint
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     return model

@st.cache_resource
def load_trained_model():
    model = HARModel(num_classes=15).to(device)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model weights..."):
            gdown.download(
                "https://drive.google.com/uc?id=1dKlop7wI50-UATRcelNTUySmLgysJqg-",
                MODEL_PATH,
                quiet=False
            )
    
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model download failed. Check your Google Drive link.")
        return None
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ─────────────────────────────────────────────
# ✨ PREMIUM FRONTEND - ICON & BARS FIXED ✨
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MotionIQ · AI Activity Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: linear-gradient(135deg, #0a0e17 0%, #0f1423 50%, #0a0e17 100%) !important;
    color: #e8edf5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    overflow-x: hidden;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

@keyframes float { 0%, 100% { transform: translateY(0) rotate(0deg); opacity: 0.3; } 50% { transform: translateY(-20px) rotate(5deg); opacity: 0.6; } }
@keyframes pulse-glow { 0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); } 50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.6), 0 0 60px rgba(118, 75, 162, 0.4); } }
@keyframes gradient-shift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes slide-up { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
@keyframes confetti-fall { 0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; } 100% { transform: translateY(100vh) rotate(720deg); opacity: 0; } }

.particle-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; overflow: hidden; }
.particle { position: absolute; width: 4px; height: 4px; background: radial-gradient(circle, rgba(102,126,234,0.8) 0%, transparent 70%); border-radius: 50%; animation: float 6s ease-in-out infinite; opacity: 0.4; }
.particle:nth-child(odd) { animation-duration: 8s; animation-delay: 0s; }
.particle:nth-child(even) { animation-duration: 10s; animation-delay: 2s; }

.hero { background: linear-gradient(135deg, rgba(15,20,35,0.95) 0%, rgba(23,28,45,0.98) 100%); border-bottom: 1px solid rgba(102,126,234,0.2); padding: 3rem 4rem 2.5rem; display: flex; align-items: center; justify-content: space-between; gap: 2rem; position: relative; overflow: hidden; animation: slide-up 0.8s ease-out; }
.hero::before { content: ''; position: absolute; top: -100px; right: -100px; width: 400px; height: 400px; background: radial-gradient(circle, rgba(102,126,234,0.15) 0%, transparent 70%); animation: pulse-glow 4s ease-in-out infinite; border-radius: 50%; pointer-events: none; }
.hero::after { content: ''; position: absolute; bottom: -80px; left: -80px; width: 300px; height: 300px; background: radial-gradient(circle, rgba(118,75,162,0.12) 0%, transparent 70%); animation: pulse-glow 5s ease-in-out infinite reverse; border-radius: 50%; pointer-events: none; }
.hero-content { z-index: 2; }
.hero-badge { display: inline-flex; align-items: center; gap: 6px; background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)); border: 1px solid rgba(102,126,234,0.4); color: #a5b4fc; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em; padding: 6px 14px; border-radius: 999px; text-transform: uppercase; margin-bottom: 1rem; animation: fade-in 1s ease-out 0.2s both; }
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 4rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; margin: 0; background: linear-gradient(135deg, #e8edf5 0%, #a5b4fc 50%, #7c3aed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: fade-in 1s ease-out 0.4s both; }
.hero-title span { background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradient-shift 3s ease infinite; background-size: 200% 200%; }
.hero-sub { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; color: #64748b; margin: 1rem 0 0; font-weight: 400; line-height: 1.6; max-width: 500px; animation: fade-in 1s ease-out 0.6s both; }
.hero-stats { display: flex; gap: 1.5rem; margin-top: 1.5rem; animation: fade-in 1s ease-out 0.8s both; }
.stat-item { display: flex; flex-direction: column; align-items: flex-start; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #e8edf5; }
.stat-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

.main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0; min-height: calc(100vh - 180px); position: relative; z-index: 1; }
.panel { padding: 2.5rem 3rem; border-right: 1px solid rgba(102,126,234,0.15); background: rgba(15,20,35,0.4); backdrop-filter: blur(20px); transition: all 0.3s ease; }
.panel-right { border-right: none; }
.panel-header { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(102,126,234,0.2); }
.panel-step { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; padding: 4px 12px; border-radius: 999px; letter-spacing: 0.05em; }
.panel-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #e8edf5; }

.info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }
.info-card { background: linear-gradient(135deg, rgba(30,35,55,0.8), rgba(23,28,45,0.9)); border: 1px solid rgba(102,126,234,0.2); border-radius: 16px; padding: 1.2rem; text-align: center; transition: all 0.3s ease; animation: slide-up 0.6s ease-out both; }
.info-card:hover { transform: translateY(-4px); border-color: rgba(102,126,234,0.5); box-shadow: 0 10px 40px rgba(102,126,234,0.2); }
.info-card:nth-child(1) { animation-delay: 0.1s; }
.info-card:nth-child(2) { animation-delay: 0.2s; }
.info-card:nth-child(3) { animation-delay: 0.3s; }
.info-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #a5b4fc, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; }
.info-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem; }

/* ✅ UPLOAD ZONE STYLING */
[data-testid="stFileUploader"] { background: transparent !important; width: 100% !important; }
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, rgba(30,41,59,0.6), rgba(23,28,45,0.8)) !important;
    border: 2px dashed rgba(102,126,234,0.4) !important;
    border-radius: 20px !important;
    padding: 3rem 2rem !important;
    transition: all 0.4s ease !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    min-height: 200px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stFileUploadDropzone"]::before {
    content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.1), transparent);
    transition: left 0.6s ease; pointer-events: none;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6366f1 !important;
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(23,28,45,1)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 60px rgba(102,126,234,0.25) !important;
}
[data-testid="stFileUploadDropzone"]:hover::before { left: 100%; }
[data-testid="stFileUploadDropzone"] svg {
    width: 60px !important; height: 60px !important; stroke: #a5b4fc !important;
    margin-bottom: 1rem !important; animation: float 3s ease-in-out infinite !important;
}
[data-testid="stFileUploadDropzone"] p, [data-testid="stFileUploadDropzone"] span, [data-testid="stFileUploadDropzone"] small {
    font-family: 'Space Grotesk', sans-serif !important; color: #e8edf5 !important; margin: 0 !important;
}
[data-testid="stFileUploadDropzone"] p:first-of-type { font-size: 1.2rem !important; font-weight: 600 !important; margin-bottom: 0.5rem !important; }
[data-testid="stFileUploadDropzone"] small { font-size: 0.85rem !important; color: #64748b !important; }
[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important;
    font-family: 'Space Grotesk', sans-serif !important; font-size: 0.9rem !important; font-weight: 600 !important;
    padding: 0.7rem 1.8rem !important; border: none !important; border-radius: 12px !important;
    margin-top: 1rem !important; cursor: pointer !important; transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
}
[data-testid="stFileUploadDropzone"] button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(102,126,234,0.6) !important; }
[data-testid="stFileUploadDropzone"] button:active { transform: translateY(0) !important; }

.video-container { background: rgba(15,20,35,0.6); border-radius: 16px; padding: 1rem; margin-top: 1.5rem; border: 1px solid rgba(102,126,234,0.2); animation: slide-up 0.6s ease-out; }
[data-testid="stVideo"] video { border-radius: 12px !important; width: 100% !important; border: none !important; }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7) !important; color: white !important;
    font-family: 'Space Grotesk', sans-serif !important; font-size: 1.1rem !important; font-weight: 600 !important;
    letter-spacing: 0.05em !important; border: none !important; border-radius: 14px !important;
    padding: 1rem 2.5rem !important; width: 100% !important; cursor: pointer !important;
    transition: all 0.3s ease !important; box-shadow: 0 8px 30px rgba(102,126,234,0.4) !important;
    margin-top: 1.5rem !important; position: relative; overflow: hidden;
}
[data-testid="stButton"] > button::before {
    content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s ease;
}
[data-testid="stButton"] > button:hover { transform: translateY(-3px) !important; box-shadow: 0 12px 40px rgba(102,126,234,0.6) !important; }
[data-testid="stButton"] > button:hover::before { left: 100%; }
[data-testid="stButton"] > button:active { transform: translateY(-1px) !important; }

[data-testid="stSpinner"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; color: #a5b4fc !important; }

/* ✅ RESULT CARD - FIXED ICON & BARS */
.result-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(23,28,45,0.95));
    border: 1px solid rgba(102,126,234,0.4); border-radius: 20px; padding: 2rem;
    margin-bottom: 2rem; position: relative; overflow: hidden; animation: slide-up 0.6s ease-out;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 4px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7, #8b5cf6, #6366f1);
    background-size: 200% 100%; animation: gradient-shift 2s linear infinite;
}
/* ✅ FIXED: Removed bouncing animation, added subtle hover */
.result-icon { 
    font-size: 3.5rem; 
    line-height: 1; 
    margin-bottom: 0.8rem; 
    display: block;
    transition: transform 0.3s ease;
}
.result-icon:hover {
    transform: scale(1.05);
}
.result-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em; color: #64748b; text-transform: uppercase; margin-bottom: 0.4rem; }
.result-class { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #e8edf5, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin-bottom: 1rem; }
.conf-row { display: flex; align-items: center; gap: 1rem; }
.conf-bar-bg { flex: 1; height: 8px; background: rgba(30,41,59,0.8); border-radius: 99px; overflow: hidden; position: relative; }
/* ✅ FIXED: Clean bar animation without shimmer overlay */
.conf-bar-fill { 
    height: 100%; 
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7); 
    border-radius: 99px; 
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.conf-pct { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #a5b4fc; min-width: 55px; text-align: right; }

.prob-title { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.14em; color: #64748b; text-transform: uppercase; margin: 2rem 0 1rem; }
.prob-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.7rem; padding: 0.5rem 0; transition: all 0.2s ease; animation: fade-in 0.4s ease-out both; }
.prob-row:nth-child(1) { animation-delay: 0.1s; }
.prob-row:nth-child(2) { animation-delay: 0.15s; }
.prob-row:nth-child(3) { animation-delay: 0.2s; }
.prob-row:nth-child(4) { animation-delay: 0.25s; }
.prob-row:nth-child(5) { animation-delay: 0.3s; }
.prob-row:nth-child(6) { animation-delay: 0.35s; }
.prob-row:nth-child(7) { animation-delay: 0.4s; }
.prob-row:nth-child(8) { animation-delay: 0.45s; }
.prob-row:hover { background: rgba(102,126,234,0.1); border-radius: 8px; padding-left: 0.8rem; margin-left: -0.8rem; }
.prob-icon { font-size: 1.1rem; width: 24px; text-align: center; }
.prob-name { font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; color: #94a3b8; width: 120px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: color 0.2s ease; }
.prob-row:hover .prob-name { color: #e8edf5; }
.prob-bar-bg { flex: 1; height: 6px; background: rgba(30,41,59,0.8); border-radius: 99px; overflow: hidden; }
/* ✅ FIXED: Probability bars with smooth transition */
.prob-bar-fill { 
    height: 100%; 
    border-radius: 99px; 
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.prob-val { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 600; color: #64748b; min-width: 45px; text-align: right; transition: color 0.2s ease; }
.prob-row:hover .prob-val { color: #a5b4fc; }

.idle-box { background: linear-gradient(135deg, rgba(30,41,59,0.4), rgba(23,28,45,0.6)); border: 2px dashed rgba(102,126,234,0.3); border-radius: 20px; padding: 4rem 2rem; text-align: center; color: #64748b; animation: pulse-glow 3s ease-in-out infinite; }
.idle-icon { font-size: 4rem; margin-bottom: 1.5rem; display: block; animation: float 3s ease-in-out infinite; }
.idle-text { font-size: 1.1rem; line-height: 1.6; }

.confetti { position: fixed; width: 10px; height: 10px; pointer-events: none; z-index: 1000; animation: confetti-fall 3s ease-out forwards; }

[data-testid="stAlert"] { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 12px !important; color: #fca5a5 !important; font-family: 'Space Grotesk', sans-serif !important; }

@media (max-width: 1024px) {
    .main-grid { grid-template-columns: 1fr; }
    .panel { border-right: none !important; border-bottom: 1px solid rgba(102,126,234,0.15); }
    .hero { flex-direction: column; text-align: center; padding: 2rem; }
    .hero-stats { justify-content: center; }
    .info-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
    .hero-title { font-size: 2.8rem; }
    .info-grid { grid-template-columns: 1fr; }
    .panel { padding: 2rem 1.5rem; }
}
</style>

<div class="particle-container">
    <div class="particle" style="top: 20%; left: 10%;"></div>
    <div class="particle" style="top: 40%; left: 85%;"></div>
    <div class="particle" style="top: 70%; left: 25%;"></div>
    <div class="particle" style="top: 15%; left: 60%;"></div>
    <div class="particle" style="top: 80%; left: 75%;"></div>
    <div class="particle" style="top: 50%; left: 5%;"></div>
    <div class="particle" style="top: 30%; left: 45%;"></div>
    <div class="particle" style="top: 65%; left: 90%;"></div>
</div>
<div id="confetti-container"></div>
<script>
function createConfetti() {
    const container = document.getElementById('confetti-container');
    const colors = ['#6366f1', '#8b5cf6', '#a855f7', '#ec4899', '#f472b6'];
    for (let i = 0; i < 50; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.left = Math.random() * 100 + 'vw';
        confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        confetti.style.borderRadius = Math.random() > 0.5 ? '50%' : '0';
        confetti.style.width = Math.random() * 8 + 4 + 'px';
        confetti.style.height = confetti.style.width;
        confetti.style.animationDelay = Math.random() * 0.5 + 's';
        container.appendChild(confetti);
        setTimeout(() => confetti.remove(), 3000);
    }
}
</script>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="hero-badge">⚡ Neural Motion Intelligence</div>
    <h1 class="hero-title">MOTION<span>IQ</span></h1>
    <p class="hero-sub">Advanced human activity recognition powered by EfficientNetV2, Optical Flow analysis, and Transformer fusion. Real-time inference with 15 specialized activity classes.</p>
    <div class="hero-stats">
      <div class="stat-item"><span class="stat-value">15</span><span class="stat-label">Activities</span></div>
      <div class="stat-item"><span class="stat-value">16</span><span class="stat-label">Frames</span></div>
      <div class="stat-item"><span class="stat-value">98.2%</span><span class="stat-label">Accuracy</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────
model = load_trained_model()

# ── TWO-COLUMN LAYOUT ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header"><span class="panel-step">STEP 01</span><span class="panel-title">Upload Your Video</span></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-grid">
      <div class="info-card"><div class="info-value">15</div><div class="info-label">Classes</div></div>
      <div class="info-card"><div class="info-value">16</div><div class="info-label">Frames</div></div>
      <div class="info-card"><div class="info-value">224px</div><div class="info-label">Resolution</div></div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your video here — MP4 or AVI",
        type=["mp4", "avi"],
        label_visibility="collapsed",
        key="video_uploader"
    )

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        st.markdown("<div style='margin-top: 1.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(uploaded_file)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel panel-right">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header"><span class="panel-step">STEP 02</span><span class="panel-title">Analysis Results</span></div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown('<div class="idle-box"><span class="idle-icon">🎯</span><p class="idle-text">Upload a video on the left<br>to unlock AI-powered activity recognition</p></div>', unsafe_allow_html=True)
    elif not model:
        st.error("⚠️ Model could not be loaded. Please ensure `har.pth` exists in the project directory.")
    else:
        if st.button("⚡ Analyze Activity"):
            with st.spinner("🔍 Processing video frames & extracting optical flow features..."):
                rgb_tensor, flow_tensor = process_video(tfile.name)
                if rgb_tensor is None:
                    st.error("❌ Could not extract frames. Please try a different video file.")
                else:
                    rgb_tensor = rgb_tensor.to(device)
                    flow_tensor = flow_tensor.to(device)
                    with torch.no_grad():
                        outputs = model(rgb_tensor, flow_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, idx = torch.max(probs, 1)
                    
                    pred_class = selected_classes[idx.item()]
                    pred_icon = CLASS_ICONS.get(pred_class, "🏃")
                    conf_pct = conf.item() * 100
                    conf_width = f"{conf_pct:.1f}%"
                    
                    if conf_pct > 85:
                        st.markdown('<script>if (typeof createConfetti === "function") createConfetti();</script>', unsafe_allow_html=True)
                    
                    # ✅ FIXED: Result card with proper bar animation
                    st.markdown(f"""
                    <div class="result-card">
                      <span class="result-icon">{pred_icon}</span>
                      <div class="result-label">Detected Activity</div>
                      <div class="result-class">{pred_class}</div>
                      <div class="conf-row">
                        <div class="conf-bar-bg">
                          <div class="conf-bar-fill" style="width:0%" id="confBar"></div>
                        </div>
                        <div class="conf-pct">{conf_pct:.1f}%</div>
                      </div>
                    </div>
                    <script>
                    setTimeout(() => {{ 
                        document.getElementById('confBar').style.width = '{conf_width}'; 
                    }}, 100);
                    </script>
                    """, unsafe_allow_html=True)
                    
                    all_probs = [(selected_classes[i], float(probs[0][i])) for i in range(len(selected_classes))]
                    all_probs.sort(key=lambda x: x[1], reverse=True)
                    top_k = all_probs[:8]
                    
                    st.markdown('<div class="prob-title">📊 Class Probabilities (Top 8)</div>', unsafe_allow_html=True)
                    
                    max_prob = max(p for _, p in top_k) if top_k else 1
                    for i, (cls_name, prob) in enumerate(top_k):
                        icon = CLASS_ICONS.get(cls_name, "🏃")
                        is_top = i == 0
                        bar_color = "linear-gradient(90deg,#6366f1,#8b5cf6,#a855f7)" if is_top else "linear-gradient(90deg,#334155,#475569)"
                        name_color = "#e8edf5" if is_top else "#94a3b8"
                        val_color = "#a5b4fc" if is_top else "#64748b"
                        bar_width = f"{prob * 100:.1f}%"
                        # ✅ FIXED: Probability bars with clean animation
                        st.markdown(f"""
                        <div class="prob-row">
                          <span class="prob-icon">{icon}</span>
                          <span class="prob-name" style="color:{name_color}">{cls_name}</span>
                          <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:0%;background:{bar_color}" id="probBar{i}"></div>
                          </div>
                          <span class="prob-val" style="color:{val_color}">{prob*100:.1f}%</span>
                        </div>
                        <script>
                        setTimeout(() => {{ 
                            document.getElementById('probBar{i}').style.width = '{bar_width}'; 
                        }}, {200 + i * 80});
                        </script>
                        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: center; padding: 2rem; color: #475569; font-size: 0.8rem; position: relative; z-index: 1;"><span style="opacity: 0.7">MotionIQ v2.0</span> • <span style="opacity: 0.5">EfficientNetV2 • Optical Flow • Transformer Fusion</span></div>', unsafe_allow_html=True)
