# load Euclid astroPT model

import torch
from astropt.model_utils import load_astropt

# Hugging Face repo ID for the Euclid VIS+NISP pretrained model
REPO_ID   = "msiudek/astroPT_euclid_VIS_NISP_model"
MODEL_DIR = "astropt/090M"          # subfolder name inside the repo
WEIGHTS   = "ckpt.pt"               # model file name

# Load the model
print("Loading AstroPT Euclid VIS+NISP model...")
loaded = load_astropt(repo_id=REPO_ID, path=MODEL_DIR, weights_filename=WEIGHTS)

# Depending on library version, this may return (model, model_args)
if isinstance(loaded, tuple):
    model, model_args = loaded
else:
    model, model_args = loaded, None

device = "cpu"                      # CUDA not available in my laptop # change for server 
model.to(device)
model.eval()

print("Model loaded successfully")
print(f"Model type : {type(model).__name__}")
if model_args:
    print("Model args :", model_args)
print("Ready for use!")

