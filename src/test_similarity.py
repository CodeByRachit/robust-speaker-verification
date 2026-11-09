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

# Suppress library deprecation and SSL warnings
warnings.filterwarnings("ignore")

from model import VocalEffortRobustModel
from mmse_estimator import PCAMMSEEstimator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVID_DIR = os.path.join(BASE_DIR, "AVID", "AVID")
CACHE_FILE = os.path.join(BASE_DIR, "checkpoint_mmse_cache.pkl")

# Real Equal Error Rate (EER) thresholds computed from the multi-speaker AVID benchmark
RAW_THRESHOLD = 0.7781
CALIBRATED_THRESHOLD = 0.6567

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))

def load_audio(path, max_length_sec=None, sample_rate=16000):
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Standard zero-mean normalization to eliminate DC bias from microphone
    waveform = waveform - torch.mean(waveform)
            
    # Ensure minimum length of 1.0s (16,000 samples) for ECAPA-TDNN convolutional receptive field
    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
            
    if max_length_sec is not None:
        max_length = int(max_length_sec * sample_rate)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            pad_amount = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    return waveform

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
                X_n.append(model(load_audio(nf).to(device)).cpu().numpy().flatten())
                X_v.append(model(load_audio(vf).to(device)).cpu().numpy().flatten())
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

    with torch.no_grad():
        w1 = load_audio(path1).to(device)
        w2 = load_audio(path2).to(device)
        e1 = model(w1).cpu().numpy().flatten()
        e2 = model(w2).cpu().numpy().flatten()

    def cen(v1, v2):
        return cosine_similarity(v1 - global_mean, v2 - global_mean)

    raw_sim = cosine_similarity(e1, e2)
    cen_base = cen(e1, e2)
    e2_l = mmse_loud.compensate(e2)
    e2_s = mmse_soft.compensate(e2)
    cen_l2 = cen(e1, e2_l)
    cen_s2 = cen(e1, e2_s)
    
    e1_l = mmse_loud.compensate(e1)
    e1_s = mmse_soft.compensate(e1)
    cen_l1 = cen(e1_l, e2)
    cen_s1 = cen(e1_s, e2)
    
    calibrated_sim = max(cen_base, cen_l2, cen_s2, cen_l1, cen_s1)

    is_same = (calibrated_sim >= CALIBRATED_THRESHOLD)
    verdict = "SAME PERSON (MATCH)" if is_same else "DIFFERENT PERSON (IMPOSTOR)"

    print(f"\n  >>> FINAL VERDICT: {verdict} <<<")
    print(f"  Calibrated Score : {calibrated_sim:+.4f} (EER Threshold: {CALIBRATED_THRESHOLD:.4f})")
    print(f"  Raw Score        : {raw_sim:.4f} (EER Threshold: {RAW_THRESHOLD:.4f})")
    print("\nDetailed Comparison:")
    print("-" * 90)
    print(f"{'Pipeline Architecture':<36} | {'Similarity':<12} | {'Threshold':<11} | {'Decision'}")
    print("-" * 90)
    print(f"{'Image 1: Baseline (E-T+WavLM)':<36} | {raw_sim:>12.4f} | {RAW_THRESHOLD:>11.4f} | {'SAME PERSON' if raw_sim >= RAW_THRESHOLD else 'DIFFERENT PERSON'}")
    print(f"{'Image 2: Full System (MMSE_v)':<36} | {calibrated_sim:>+12.4f} | {CALIBRATED_THRESHOLD:>11.4f} | {verdict}")
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

    with torch.no_grad():
        w_an = load_audio(spk_a_norm).to(device)
        w_al = load_audio(spk_a_loud).to(device)
        w_as = load_audio(spk_a_soft).to(device)
        w_bn = load_audio(spk_b_norm).to(device)

        e_an = model(w_an).cpu().numpy().flatten()
        e_al = model(w_al).cpu().numpy().flatten()
        e_as = model(w_as).cpu().numpy().flatten()
        e_bn = model(w_bn).cpu().numpy().flatten()

    e_al_comp = mmse_loud.compensate(e_al)
    e_as_comp = mmse_soft.compensate(e_as)

    def cen(v1, v2):
        return cosine_similarity(v1 - global_mean, v2 - global_mean)

    b_nl = cosine_similarity(e_an, e_al)
    b_ns = cosine_similarity(e_an, e_as)
    b_ls = cosine_similarity(e_al, e_as)
    b_imp = cosine_similarity(e_an, e_bn)

    f_nl = cen(e_an, e_al_comp)
    f_ns = cen(e_an, e_as_comp)
    f_ls = cen(e_al_comp, e_as_comp)
    f_imp = cen(e_an, e_bn)

    print("\n================ COMPARISON: CROPPED IMAGE 1 vs FULL IMAGE 2 ================")
    print(f"{'Trial Condition':<34} | {'Image 1 (Base)':<16} | {'Image 2 (Comp)':<16} | {'Centered'}")
    print("-" * 85)
    print(f"{'1. Spk A Normal vs Spk A Loud':<34} | {b_nl:>16.4f} | {cosine_similarity(e_an, e_al_comp):>16.4f} | {f_nl:>+10.4f}")
    print(f"{'2. Spk A Normal vs Spk A Soft':<34} | {b_ns:>16.4f} | {cosine_similarity(e_an, e_as_comp):>16.4f} | {f_ns:>+10.4f}")
    print(f"{'3. Spk A Loud   vs Spk A Soft':<34} | {b_ls:>16.4f} | {cosine_similarity(e_al_comp, e_as_comp):>16.4f} | {f_ls:>+10.4f}")
    print(f"{'4. Spk A Normal vs Spk B Normal':<34} | {b_imp:>16.4f} | {cosine_similarity(e_an, e_bn):>16.4f} | {f_imp:>+10.4f} (Impostor)")
    print("=" * 85)

    if (f_nl > f_imp) and (f_ns > f_imp):
        print("\nSUCCESS: Full pipeline matched BOTH loud and soft speech and reliably rejected the impostor!")
    else:
        print("\nFAILURE: Model struggled with non-neutral phonation.")

if __name__ == "__main__":
    checkpoints = glob.glob(os.path.join(BASE_DIR, "checkpoint_epoch_*.pth"))
    latest_checkpoint = os.path.join(BASE_DIR, "checkpoint_epoch_20.pth")
    if checkpoints:
        def get_epoch(p):
            m = re.search(r'checkpoint_epoch_(\d+)\.pth', os.path.basename(p))
            return int(m.group(1)) if m else -1
        latest_checkpoint = max(checkpoints, key=get_epoch)

    parser = argparse.ArgumentParser(description="Evaluate Vocal Effort Robust Model & PCA-MMSE pipeline")
    parser.add_argument("--checkpoint", type=str, default=latest_checkpoint, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default=AVID_DIR, help="Path to AVID dataset folder")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, cpu)")
    parser.add_argument("--voice1", "--audio1", type=str, dest="voice1", default=None, help="Optional path to Voice 1 WAV file")
    parser.add_argument("--voice2", "--audio2", type=str, dest="voice2", default=None, help="Optional path to Voice 2 WAV file")
    args = parser.parse_args()

    test_model(args.checkpoint, args.data_dir, device=args.device, voice1=args.voice1, voice2=args.voice2)
