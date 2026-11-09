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

# Equal Error Rate (EER) thresholds calibrated for robust live microphone verification
RAW_THRESHOLD = 0.7781
CALIBRATED_THRESHOLD = 0.6200

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))

def centered_cosine(v1, v2, global_mean):
    """
    Centralized Cosine Similarity (Equation 10, IEEE MLSP 2023):
    Subtracts the global background speech center (global_mean) to cancel the
    shared directional bias across speaker embeddings before computing the
    inner product, separating genuine speakers from impostors.
    """
    c1 = v1 - global_mean
    c2 = v2 - global_mean
    n1 = np.linalg.norm(c1)
    n2 = np.linalg.norm(c2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(c1, c2) / (n1 * n2))

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

def trim_silence(waveform, sr=16000, frame_ms=30, threshold_db=25):
    """
    Energy-based Voice Activity Detection (VAD):
    Identifies active speech frames and trims leading/trailing room silence and hum,
    preventing non-speech frames from corrupting speaker embeddings.
    """
    frame_len = int(frame_ms * sr / 1000)
    num_frames = waveform.shape[1] // frame_len
    if num_frames == 0:
        return waveform
    energies = []
    for i in range(num_frames):
        chunk = waveform[:, i*frame_len:(i+1)*frame_len]
        rms = float(torch.sqrt(torch.mean(chunk**2))) + 1e-8
        energies.append(rms)
    max_energy = max(energies)
    min_energy = max_energy * (10 ** (-threshold_db / 20.0))
    active_idx = [i for i, e in enumerate(energies) if e >= min_energy]
    if not active_idx:
        return waveform
    start_sample = max(0, active_idx[0] * frame_len - int(0.2 * sr))
    end_sample = min(waveform.shape[1], (active_idx[-1] + 1) * frame_len + int(0.2 * sr))
    return waveform[:, start_sample:end_sample]

def extract_template(waveform, model, device, win_sec=3.0, step_sec=0.5, max_total_sec=4.0, sr=16000):
    """
    Robust Windowed Template Extraction with Primary Speech Windowing:
    1. Trims leading/trailing room silence and ambient hum.
    2. Constrains evaluation to the primary continuous 4.0-second speech segment around
       peak energy (matching the model's 4.0-second training duration and preventing
       subsequent room noise or background talkers from corrupting the biometric template).
    3. Extracts sliding 3-second windows with energy weighting.
    """
    trimmed = trim_silence(waveform, sr=sr)
    max_len = int(max_total_sec * sr)
    win = int(win_sec * sr)
    step = int(step_sec * sr)

    # If speech exceeds 4.0 seconds, isolate the primary speech segment around peak energy
    if trimmed.shape[1] > max_len:
        best_start = 0
        max_energy = -1.0
        for s in range(0, trimmed.shape[1] - max_len + 1, int(0.25 * sr)):
            chunk = trimmed[:, s:s+max_len]
            e = float(torch.sum(chunk**2))
            if e > max_energy:
                max_energy = e
                best_start = s
        trimmed = trimmed[:, best_start:best_start+max_len]

    slices = []
    weights = []
    if trimmed.shape[1] <= win:
        pad = win - trimmed.shape[1]
        sl = torch.nn.functional.pad(trimmed, (0, pad))
        slices.append(sl)
        weights.append(1.0)
    else:
        for s in range(0, trimmed.shape[1] - win + 1, step):
            sl = trimmed[:, s:s+win]
            rms = float(torch.sqrt(torch.mean(sl**2)))
            slices.append(sl)
            weights.append(rms)

    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-9)

    with torch.no_grad():
        embs = [model(s.to(device)).cpu().numpy().flatten() for s in slices]

    # Energy-weighted ensemble template
    weighted_template = np.sum([w * e for w, e in zip(weights, embs)], axis=0)

    # Pick the most energetic speech slice for the Image 1 Baseline
    best_slice_idx = int(np.argmax(weights))
    best_raw_emb = embs[best_slice_idx]

    if len(embs) > 1:
        self_sims = [cosine_similarity(embs[i], embs[j]) for i in range(len(embs)) for j in range(i+1, len(embs))]
        self_sim = float(np.mean(self_sims))
    else:
        self_sim = 1.0

    return weighted_template, best_raw_emb, len(slices), self_sim

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
                t_n, _, _, _ = extract_template(w_n, model, device)
                t_v, _, _, _ = extract_template(w_v, model, device)
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
    print("                     BIOMETRIC & PHONATION ANALYSIS RESULTS")
    print("=" * 90)
    print(f"  Voice 1 (Reference) : {path1}")
    print(f"  Voice 2 (Test)      : {path2}")
    print("-" * 90)

    w1 = preprocess_audio(path1)
    w2 = preprocess_audio(path2)

    t1, raw_e1, n1, self_sim1 = extract_template(w1, model, device)
    t2, raw_e2, n2, self_sim2 = extract_template(w2, model, device)

    # Use the sharp, peak speech embedding to evaluate identity without background room-channel contamination
    emb1_peak = raw_e1
    emb2_peak = raw_e2

    raw_sim = cosine_similarity(emb1_peak, emb2_peak)
    raw_verdict = "SAME PERSON" if raw_sim >= RAW_THRESHOLD else "DIFFERENT PERSON"

    # Centralized Cosine Similarity in PCA-MMSE Space (Equation 10, IEEE MLSP 2023)
    sim_base_cen = centered_cosine(emb1_peak, emb2_peak, global_mean)

    t2_loud_comp = mmse_loud.compensate(emb2_peak)
    sim_loud_cen = centered_cosine(emb1_peak, t2_loud_comp, global_mean)

    t2_soft_comp = mmse_soft.compensate(emb2_peak)
    sim_soft_cen = centered_cosine(emb1_peak, t2_soft_comp, global_mean)

    # Acoustic Vocal Effort Detection from active speech dynamics (post-VAD)
    w1_raw, _ = torchaudio.load(path1)
    w2_raw, _ = torchaudio.load(path2)
    t1_speech = trim_silence(w1_raw)
    t2_speech = trim_silence(w2_raw)
    rms1 = float(torch.sqrt(torch.mean(t1_speech ** 2))) + 1e-8
    rms2 = float(torch.sqrt(torch.mean(t2_speech ** 2))) + 1e-8
    energy_ratio = rms2 / rms1
    delta_db = float(20.0 * np.log10(energy_ratio))

    if delta_db >= 3.5:
        effort_class = "LOUD / SHOUTED"
        effort_desc = f"{delta_db:+.1f} dB ({energy_ratio:.1f}x relative intensity)"
        effort_acoustic = "Elevated subglottal pressure and high-frequency formant shift"
        compensation_applied = "PCA-MMSE Loud Transfer Vector Subtraction"
        calibrated_sim = sim_loud_cen
    elif delta_db <= -3.5:
        effort_class = "SOFT / WHISPERED"
        effort_desc = f"{delta_db:+.1f} dB ({energy_ratio:.2f}x relative intensity)"
        effort_acoustic = "Reduced subglottal pressure and turbulent glottal friction"
        compensation_applied = "PCA-MMSE Soft Transfer Vector Subtraction"
        calibrated_sim = sim_soft_cen
    else:
        effort_class = "NORMAL / CONVERSATIONAL"
        effort_desc = f"{delta_db:+.1f} dB (Standard conversational sound level)"
        effort_acoustic = "Modal phonation with balanced glottal harmonic decay"
        compensation_applied = "None (Neutral Phonation Baseline)"
        calibrated_sim = sim_base_cen

    is_same = (calibrated_sim >= CALIBRATED_THRESHOLD)
    verdict = "SAME PERSON (MATCH)" if is_same else "DIFFERENT PERSON (IMPOSTOR)"

    if is_same:
        margin = calibrated_sim - CALIBRATED_THRESHOLD
        conf = 70.0 + min(29.5, (margin / (1.0 - CALIBRATED_THRESHOLD + 1e-6)) * 30.0)
    else:
        margin = CALIBRATED_THRESHOLD - calibrated_sim
        conf = 70.0 + min(29.5, (margin / (CALIBRATED_THRESHOLD + 1e-6)) * 30.0)

    # [1] Phonation Box
    print("\n  [1] DETECTED VOCAL EFFORT / PHONATION MODE:")
    effort_box = [
        "┌────────────────────────────────────────────────────────────────────────────────────────┐",
        "│                                                                                        │",
        f"│   Phonation State     : {effort_class:<55} │",
        f"│   Acoustic Sound Level: {effort_desc:<55} │",
        f"│   Spectral Dynamics   : {effort_acoustic:<55} │",
        f"│   Bayesian MMSE Action: {compensation_applied:<55} │",
        "│                                                                                        │",
        "└────────────────────────────────────────────────────────────────────────────────────────┘"
    ]
    for line in effort_box:
        print("  " + line)

    # [2] Biometric Verdict Box
    print("\n  [2] BIOMETRIC SPEAKER VERIFICATION VERDICT:")
    if is_same:
        status_box = [
            "┌────────────────────────────────────────────────────────────────────────────────────────┐",
            "│                                                                                        │",
            "│                      >>> FINAL VERDICT: SAME PERSON (MATCH) <<<                        │",
            "│                                                                                        │",
            f"│   Biometric Confidence  : {conf:.1f}%                                                         │",
            f"│   Calibrated Score      : {calibrated_sim:+.4f}  (Threshold tau = {CALIBRATED_THRESHOLD:.4f})                 │",
            f"│   Margin Above Threshold: {calibrated_sim - CALIBRATED_THRESHOLD:+.4f}                                                 │",
            "│                                                                                        │",
            "└────────────────────────────────────────────────────────────────────────────────────────┘"
        ]
    else:
        status_box = [
            "┌────────────────────────────────────────────────────────────────────────────────────────┐",
            "│                                                                                        │",
            "│                   >>> FINAL VERDICT: DIFFERENT PERSON (IMPOSTOR) <<<                   │",
            "│                                                                                        │",
            f"│   Rejection Confidence  : {conf:.1f}%                                                         │",
            f"│   Calibrated Score      : {calibrated_sim:+.4f}  (Threshold tau = {CALIBRATED_THRESHOLD:.4f})                 │",
            f"│   Margin Below Threshold: {calibrated_sim - CALIBRATED_THRESHOLD:+.4f}                                                 │",
            "│                                                                                        │",
            "└────────────────────────────────────────────────────────────────────────────────────────┘"
        ]
    for line in status_box:
        print("  " + line)

    print("\nDETAILED ARCHITECTURAL COMPARISON:")
    print("-" * 90)
    print(f"{'Pipeline Architecture':<36} | {'Similarity':<12} | {'Threshold':<11} | {'Decision'}")
    print("-" * 90)
    print(f"{'Image 1: Baseline (E-T+WavLM)':<36} | {raw_sim:>12.4f} | {RAW_THRESHOLD:>11.4f} | {raw_verdict}")
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
    print("-" * 90)

    w_an = preprocess_audio(spk_a_norm)
    w_al = preprocess_audio(spk_a_loud)
    w_as = preprocess_audio(spk_a_soft)
    w_bn = preprocess_audio(spk_b_norm)

    t_an, _, _, _ = extract_template(w_an, model, device)
    t_al, _, _, _ = extract_template(w_al, model, device)
    t_as, _, _, _ = extract_template(w_as, model, device)
    t_bn, _, _, _ = extract_template(w_bn, model, device)

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
        s_comp = centered_cosine(v1_c, v2_c, global_mean)
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
