import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify, render_template

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

class OptimizedCKDModel(nn.Module):
    def __init__(self, input_size=10, embed_dim=32, dropout_rate=0.59):
        super(OptimizedCKDModel, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.W = nn.Parameter(torch.Tensor(1, input_size, embed_dim))
        self.b = nn.Parameter(torch.Tensor(1, input_size, embed_dim))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        nn.init.zeros_(self.b)
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True, dropout=dropout_rate/2)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        tokens = x_expanded * self.W + self.b
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        out = self.classifier(tokens)
        return out

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nn_model = OptimizedCKDModel()
nn_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'ft_transformer_ckd_sec.pth'), map_location='cpu', weights_only=True))
nn_model.eval()

scaler = joblib.load(os.path.join(BASE_DIR, 'ckd_scaler.pkl'))
xgb_model = joblib.load(os.path.join(BASE_DIR, 'xgboost_ckd.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        def safe_float(val):
            try:
                return float(val) if val != "" else 0.0
            except:
                return 0.0

        features = [
            safe_float(data.get('FastingBloodSugar')),
            safe_float(data.get('HbA1c')),
            safe_float(data.get('SerumCreatinine')),
            safe_float(data.get('BUNLevels')),
            safe_float(data.get('GFR')),
            safe_float(data.get('ProteinInUrine')),
            safe_float(data.get('Sodium')),
            safe_float(data.get('Hemoglobin')),
            safe_float(data.get('CholesterolTotal')),
            safe_float(data.get('CholesterolHDL'))
        ]
        
        raw_input = np.array([features])
        scaled_input = scaler.transform(raw_input)
        
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        with torch.no_grad():
            logits = nn_model(input_tensor)
            nn_prob = torch.sigmoid(logits).item()
            
        xgb_prob = float(xgb_model.predict_proba(scaled_input)[:, 1][0])
        
        final_prob = (nn_prob + xgb_prob) / 2
        risk_level = "Жоғары қауіп" if final_prob >= 0.30 else "Төмен қауіп"
        
        return jsonify({
            'probability': final_prob,
            'nn': nn_prob,
            'xgb': xgb_prob,
            'risk': risk_level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
