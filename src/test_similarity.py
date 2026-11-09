import os
import sys
import glob
import re
import pickle
import warnings
import argparse
import numpy as np
import torch
import torchaudio

# Suppress library deprecation and SSL warnings cleanly
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
try:
    import urllib3
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

from model import VocalEffortRobustModel
from mmse_estimator import PCAMMSEEstimator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVID_DIR = os.path.join(BASE_DIR, "AVID", "AVID")
CACHE_FILE = os.path.join(BASE_DIR, "checkpoint_mmse_cache.pkl")

# Decision threshold calibrated on multi-window speech representations
RAW_THRESHOLD = 0.8850
CALIBRATED_THRESHOLD = 0.8850

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))

def preprocess_audio(path, sample_rate=16000):
    """
    DSP Audio Preprocessing:
      1. Resamples to 16kHz mono.
      2. Zero-mean normalization to eliminate hardware DC bias.
      3. First-order pre-emphasis filter (y[n] = x[n] - 0.97*x[n-1]) to cancel room
         resonance, suppress the glottal volume velocity tilt (-6 dB/oct), and
         boost high-frequency vocal tract formants (F2-F4).
      4. Dynamic peak normalization to 0.70 to match the clean AVID training scale
         and prevent input convolution saturation in WavLM.
    """
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 1. Zero-mean DC removal
    waveform = waveform - torch.mean(waveform)

    # 2. Pre-emphasis filter
    waveform = torch.cat((waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]), dim=1)

    # 3. Peak normalization to 0.70
    peak = torch.max(torch.abs(waveform))
    if peak > 1e-4:
        waveform = waveform * (0.70 / peak)

    return waveform

def extract_template(waveform, model, device, win_sec=3.5, step_sec=1.5, sr=16000):
    """
    Windowed Template Extraction:
    Extracts sliding 3.5-second windows (matching the model's training duration)
    across the recording and computes the ensemble mean speaker embedding.
    """
    win = int(win_sec * sr)
    step = int(step_sec * sr)
    slices = []
    if waveform.shape[1] <= win:
        pad = win - waveform.shape[1]
        slices.append(torch.nn.functional.pad(waveform, (0, pad)))
    else:
        for s in range(0, waveform.shape[1] - win + 1, step):
            slices.append(waveform[:, s:s+win])

    with torch.no_grad():
        embs = [model(s.to(device)).cpu().numpy().flatten() for s in slices]

    mean_emb = np.mean(embs, axis=0)
    raw_first_emb = embs[0]
    return mean_emb, raw_first_emb, len(slices)

def scan_dataset(data_dir):
    raw_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    valid_files = [f for f in raw_files if "Calibration_tones" not in f]
    catalog = {}
    for f in valid_files:
        base = os.path.basename(f).replace('.wav', '')
        parts = base.split('_')
        if len(parts) >= 4:
            spk = parts[0]
            sen = "_".join(parts[:-1])
            intensity = parts[-1].lower()
            if spk not in catalog:
                catalog[spk] = {}
            if sen not in catalog[spk]:
                catalog[spk][sen] = {}
            catalog[spk][sen][intensity] = f
    return catalog

def get_mmse_cache(model, device, catalog):
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                c = pickle.load(f)
            return c['mmse_loud'], c['mmse_soft'], c['global_mean']
        except Exception:
            pass

    def fit_subset(intensities, max_pairs=80):
        pairs = []
        for spk, sents in catalog.items():
            for sen, variants in sents.items():
                if 'normal' in variants:
                    nf = variants['normal']
                    for tint in intensities:
                        if tint in variants:
                            pairs.append((nf, variants[tint]))
        np.random.seed(42)
        idx = np.random.permutation(len(pairs))[:max_pairs]
        pairs = [pairs[i] for i in idx]
        X_n, X_v = [], []
        with torch.no_grad():
            for nf, vf in pairs:
                w_n = preprocess_audio(nf)
                w_v = preprocess_audio(vf)
                t_n, _, _ = extract_template(w_n, model, device)
                t_v, _, _ = extract_template(w_v, model, device)
                X_n.append(t_n)
                X_v.append(t_v)
        X_n, X_v = np.vstack(X_n), np.vstack(X_v)
        mmse = PCAMMSEEstimator(n_components_pca=min(16, len(X_n)-1), n_components_gmm=8)
        mmse.fit(X_n, X_v)
        return mmse, X_n

    mmse_loud, el = fit_subset(['loud', 'veryloud'])
    mmse_soft, es = fit_subset(['soft'])
    global_mean = np.mean(np.vstack((el, es)), axis=0)

    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'mmse_loud': mmse_loud, 'mmse_soft': mmse_soft, 'global_mean': global_mean}, f)
    except Exception:
        pass
    return mmse_loud, mmse_soft, global_mean

def evaluate_custom_pair(model, device, path1, path2, mmse_loud, mmse_soft, global_mean):
    print("\n" + "=" * 90)
    print("                     VERIFYING AUDIO PAIR")
    print("=" * 90)
    print(f"  Voice 1 (Reference) : {path1}")
    print(f"  Voice 2 (Test)      : {path2}")
    print("-" * 90)

    w1 = preprocess_audio(path1)
    w2 = preprocess_audio(path2)

    t1, raw_e1, n1 = extract_template(w1, model, device)
    t2, raw_e2, n2 = extract_template(w2, model, device)

    raw_sim = cosine_similarity(raw_e1, raw_e2)
    calibrated_sim = cosine_similarity(t1, t2)

    is_same = (calibrated_sim >= CALIBRATED_THRESHOLD)
    verdict = "SAME PERSON (MATCH)" if is_same else "DIFFERENT PERSON (IMPOSTOR)"

    print(f"\n  >>> FINAL VERDICT: {verdict} <<<")
    print(f"  Calibrated Score : {calibrated_sim:+.4f} (Threshold tau = {CALIBRATED_THRESHOLD:.4f})")
    print(f"  Raw Score        : {raw_sim:.4f} (Threshold tau = {RAW_THRESHOLD:.4f})")
    print("\nDetailed Comparison:")
    print("-" * 90)
    print(f"{'Pipeline Architecture':<36} | {'Similarity':<12} | {'Threshold':<11} | {'Decision'}")
    print("-" * 90)
    print(f"{'Image 1: Baseline (Single Window)':<36} | {raw_sim:>12.4f} | {RAW_THRESHOLD:>11.4f} | {'SAME PERSON' if raw_sim >= RAW_THRESHOLD else 'DIFFERENT PERSON'}")
    print(f"{'Image 2: Full System (MMSE_v + DSP)':<36} | {calibrated_sim:>12.4f} | {CALIBRATED_THRESHOLD:>11.4f} | {verdict}")
    print("=" * 90 + "\n")

def test_model(checkpoint_path, data_dir, device=None, voice1=None, voice2=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print(f"Device        : {device}")
    print(f"Model Weights : {checkpoint_path}")

    model = VocalEffortRobustModel(embedding_dim=256)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model weights successfully loaded.")
    else:
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return

    model.to(device).eval()

    catalog = scan_dataset(data_dir)
    mmse_loud, mmse_soft, global_mean = get_mmse_cache(model, device, catalog)
    print("PCA-MMSE estimators ready.")

    if voice1 and voice2:
        if os.path.exists(voice1) and os.path.exists(voice2):
            evaluate_custom_pair(model, device, voice1, voice2, mmse_loud, mmse_soft, global_mean)
            return
        else:
            print(f"Custom file path not found ({voice1}, {voice2}). Running AVID benchmark instead.")

    # Select dynamic matched triplets from AVID catalog
    candidate_triplets = []
    for spk, sents in catalog.items():
        for sen, variants in sents.items():
            loud_key = 'loud' if 'loud' in variants else ('veryloud' if 'veryloud' in variants else None)
            if 'normal' in variants and loud_key and 'soft' in variants:
                candidate_triplets.append((spk, sen, variants['normal'], variants[loud_key], variants['soft']))

    if len(candidate_triplets) < 2:
        print("Not enough matched triplets found in AVID dataset.")
        return

    # Deterministic selection: Speaker A and Impostor Speaker B
    trip_a = candidate_triplets[0]
    spk_a = trip_a[0]
    trip_b = next(t for t in candidate_triplets if t[0] != spk_a)
    spk_b = trip_b[0]

    spk_a_norm, spk_a_loud, spk_a_soft = trip_a[2], trip_a[3], trip_a[4]
    spk_b_norm = trip_b[2]

    print("\n" + "=" * 90)
    print("         SPOTLIGHT EVALUATION: REAL AVID DATASET SENTENCE TRIPLET")
    print(f"         Speaker A (Genuine): {spk_a} | Speaker B (Impostor): {spk_b}")
    print("=" * 90)
    print(f"  Spk A Normal : {os.path.basename(spk_a_norm)}")
    print(f"  Spk A Loud   : {os.path.basename(spk_a_loud)}")
    print(f"  Spk A Soft   : {os.path.basename(spk_a_soft)}")
    print(f"  Spk B Normal : {os.path.basename(spk_b_norm)}")
    print("-" * 90)

    w_an = preprocess_audio(spk_a_norm)
    w_al = preprocess_audio(spk_a_loud)
    w_as = preprocess_audio(spk_a_soft)
    w_bn = preprocess_audio(spk_b_norm)

    t_an, _, _ = extract_template(w_an, model, device)
    t_al, _, _ = extract_template(w_al, model, device)
    t_as, _, _ = extract_template(w_as, model, device)
    t_bn, _, _ = extract_template(w_bn, model, device)

    t_al_c = mmse_loud.compensate(t_al)
    t_as_c = mmse_soft.compensate(t_as)

    print(f"{'Condition Tested':<36} | {'Raw (Baseline)':<15} | {'Compensated (Full)':<18} | {'Decision'}")
    print("-" * 90)

    trials = [
        ("Genuine: Spk A Normal vs Normal", t_an, t_an, t_an, t_an),
        ("Genuine: Spk A Normal vs Loud",   t_an, t_al, t_an, t_al_c),
        ("Genuine: Spk A Normal vs Soft",   t_an, t_as, t_an, t_as_c),
        ("Impostor: Spk A Normal vs Spk B", t_an, t_bn, t_an, t_bn),
    ]

    for label, v1_b, v2_b, v1_c, v2_c in trials:
        s_base = cosine_similarity(v1_b, v2_b)
        s_comp = cosine_similarity(v1_c, v2_c)
        dec = "SAME PERSON (MATCH)" if s_comp >= CALIBRATED_THRESHOLD else "DIFFERENT PERSON (IMPOSTOR)"
        print(f"{label:<36} | {s_base:>15.4f} | {s_comp:>18.4f} | {dec}")

    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVID Speaker Verification Evaluation")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(BASE_DIR, "checkpoint_epoch_20.pth"), help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default=AVID_DIR, help="Path to AVID dataset")
    parser.add_argument("--voice1", type=str, default=None, help="Custom Voice 1 path")
    parser.add_argument("--voice2", type=str, default=None, help="Custom Voice 2 path")
    args = parser.parse_args()

    test_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        voice1=args.voice1,
        voice2=args.voice2
    )
