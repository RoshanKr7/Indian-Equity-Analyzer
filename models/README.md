# Pretrained Weights Directory

This directory stores GPU-trained model checkpoints from Kaggle for local CPU inference.

## Current Pretrained Models:

1. **Temporal Fusion Transformer (TFT)**
   - `tft_weights.pth` — PyTorch state dict for TFT
   - `tft_config.json` — TFT architecture configuration
   - `feature_scaler.pkl` — RobustScaler fitted during training

2. **N-HiTS (Neural Hierarchical Interpolation for Time Series)**
   - `nhits_weights.pth` — PyTorch state dict for N-HiTS
   - `nhits_config.json` — N-HiTS multi-horizon configuration

3. **FinancialBERT-Indian (Domain Fine-Tuned Sentiment)**
   - Directory: `finbert_indian/`
     - `config.json`
     - `model.safetensors` (or `pytorch_model.bin`)
     - `tokenizer.json` (Hugging Face Fast Tokenizer with full vocabulary)
     - `tokenizer_config.json`
     - `fine_tune_meta.json`

---

## Automatic Detection & Fallbacks:
- If `tft_weights.pth` is present → TFT is used as the primary sequential model.
- If `nhits_weights.pth` is present → N-HiTS is used (replaces Prophet, 50x faster).
- If `finbert_indian/` is present → FinancialBERT-Indian is loaded for news sentiment.
- If any weight file is absent, the ensemble automatically falls back to local models.
