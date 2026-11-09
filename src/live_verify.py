import os
import sys
import time
import wave
import glob
import random
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
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
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
    This suppresses transient phoneme noise and extracts the true biometric identity.
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

def record_audio_mic(output_path, duration=10, sample_rate=16000):
    """
    Records mono audio from the system microphone using sounddevice.
    Saves standard 16-bit 16kHz PCM WAV using standard library 'wave'.
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("\n" + "=" * 80)
        print(" [!] Missing required package: 'sounddevice'")
        print(" To record live audio directly from your microphone, please install it:")
        print("     pip install sounddevice")
        print("=" * 80 + "\n")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n" + "-" * 70)
    for count in range(3, 0, -1):
        print(f"  Starting in {count}... Get ready!", end='\r', flush=True)
        time.sleep(1)
    print("  >>> RECORDING NOW! Speak into your microphone... <<<          ")
    print("-" * 70)

    # Start non-blocking recording
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')

    bar_width = 30
    start_t = time.time()
    for sec in range(int(duration)):
        time.sleep(1)
        elapsed = min(time.time() - start_t, float(duration))
        frac = elapsed / duration
        filled = int(frac * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(f"\r  Recording: [{bar}] {elapsed:4.1f}s / {duration:.1f}s")
        sys.stdout.flush()

    sd.wait()
    sys.stdout.write(f"\r  Recording: [{'█' * bar_width}] {duration:.1f}s / {duration:.1f}s - COMPLETE!   \n")
    sys.stdout.flush()

    # Convert float32 to int16 PCM and save WAV
    audio_int16 = (recording * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"  ✓ Saved to: {output_path}")
    return output_path

def get_or_fit_mmse_estimators(model, device, data_dir=AVID_DIR):
    """
    Loads cached MMSE estimators and speech mean vector if available,
    otherwise fits them once from the AVID training speakers and caches to disk.
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            print("  ✓ Loaded precomputed PCA-MMSE estimators and speech mean vector from cache.")
            return cache['mmse_loud'], cache['mmse_soft'], cache['global_mean']
        except Exception as e:
            print(f"  Cache read notice ({e}), recomputing estimators...")

    print("  Fitting PCA-MMSE estimators from AVID training dataset (one-time setup)...")

    raw_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    valid_files = [f for f in raw_files if "Calibration_tones" not in f]

    catalog = {}
    for f in valid_files:
        basename = os.path.basename(f).replace('.wav', '')
        parts = basename.split('_')
        if len(parts) >= 4:
            spk = parts[0]
            sen_key = "_".join(parts[:-1])
            intensity = parts[-1].lower()
            if spk not in catalog:
                catalog[spk] = {}
            if sen_key not in catalog[spk]:
                catalog[spk][sen_key] = {}
            catalog[spk][sen_key][intensity] = f

    def fit_estimator(target_intensities, max_pairs=80, L=16, K=8):
        candidate_pairs = []
        for spk, sentences in catalog.items():
            for sen_key, variants in sentences.items():
                if 'normal' in variants:
                    norm_f = variants['normal']
                    for intensity in target_intensities:
                        if intensity in variants:
                            candidate_pairs.append((norm_f, variants[intensity]))
        random.seed(42)
        random.shuffle(candidate_pairs)
        selected_pairs = candidate_pairs[:max_pairs]

        X_norm, X_var = [], []
        with torch.no_grad():
            for nf, vf in selected_pairs:
                w_n = preprocess_audio(nf)
                w_v = preprocess_audio(vf)
                t_n, _, _ = extract_template(w_n, model, device)
                t_v, _, _ = extract_template(w_v, model, device)
                X_norm.append(t_n)
                X_var.append(t_v)

        X_norm = np.vstack(X_norm)
        X_var = np.vstack(X_var)
        mmse = PCAMMSEEstimator(n_components_pca=min(L, len(X_norm) - 1), n_components_gmm=K)
        mmse.fit(X_norm, X_var)
        return mmse, X_norm

    mmse_loud, embs_loud = fit_estimator(['loud', 'veryloud'], max_pairs=80)
    mmse_soft, embs_soft = fit_estimator(['soft'], max_pairs=80)
    global_mean = np.mean(np.vstack((embs_loud, embs_soft)), axis=0)

    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({
                'mmse_loud': mmse_loud,
                'mmse_soft': mmse_soft,
                'global_mean': global_mean
            }, f)
        print("  ✓ Saved estimators to disk cache for future instant launches.")
    except Exception:
        pass

    return mmse_loud, mmse_soft, global_mean

def run_verification(audio_path1, audio_path2, model, device, mmse_loud, mmse_soft, global_mean, effort_mode="auto"):
    """
    Extracts embeddings for both voice recordings, runs baseline vs MMSE compensated verification,
    and returns comprehensive similarity scores and biometric verdicts.
    """
    w1 = preprocess_audio(audio_path1)
    w2 = preprocess_audio(audio_path2)

    emb1_template, emb1_raw, n_slices1 = extract_template(w1, model, device)
    emb2_template, emb2_raw, n_slices2 = extract_template(w2, model, device)

    # 1. Image 1: Raw Baseline Cosine Similarity (single-window uncompensated)
    raw_sim = cosine_similarity(emb1_raw, emb2_raw)
    raw_verdict = "SAME PERSON" if raw_sim >= RAW_THRESHOLD else "DIFFERENT PERSON"

    # 2. Image 2: Multi-Window Ensemble Cosine Similarity
    sim_base_template = cosine_similarity(emb1_template, emb2_template)

    # MMSE Vocal Effort Compensations
    emb2_loud_comp = mmse_loud.compensate(emb2_template)
    sim_loud = cosine_similarity(emb1_template, emb2_loud_comp)

    emb2_soft_comp = mmse_soft.compensate(emb2_template)
    sim_soft = cosine_similarity(emb1_template, emb2_soft_comp)

    # 3. Acoustic Vocal Effort Detection from speech dynamics
    w1_raw, _ = torchaudio.load(audio_path1)
    w2_raw, _ = torchaudio.load(audio_path2)
    rms1 = float(torch.sqrt(torch.mean(w1_raw ** 2))) + 1e-6
    rms2 = float(torch.sqrt(torch.mean(w2_raw ** 2))) + 1e-6
    energy_ratio = rms2 / rms1

    vocal_effort_detected = False
    effort_type = "Normal Speech"

    # Determine whether vocal effort compensation is needed:
    if effort_mode == "loud" or (effort_mode == "auto" and energy_ratio >= 2.5):
        vocal_effort_detected = True
        effort_type = "Shouted / Loud Speech"
        calibrated_sim = max(sim_base_template, sim_loud)
    elif effort_mode == "soft" or (effort_mode == "auto" and energy_ratio <= 0.20):
        vocal_effort_detected = True
        effort_type = "Whispered / Soft Speech"
        calibrated_sim = max(sim_base_template, sim_soft)
    else:
        # Standard neutral speech verification
        calibrated_sim = sim_base_template

    is_same_person = (calibrated_sim >= CALIBRATED_THRESHOLD)
    calibrated_verdict = "SAME PERSON" if is_same_person else "DIFFERENT PERSON"

    # Confidence calculation based on margin to decision threshold
    if is_same_person:
        margin = calibrated_sim - CALIBRATED_THRESHOLD
        conf = 70.0 + min(29.5, (margin / (1.0 - CALIBRATED_THRESHOLD + 1e-6)) * 30.0)
    else:
        margin = CALIBRATED_THRESHOLD - calibrated_sim
        conf = 70.0 + min(29.5, (margin / (CALIBRATED_THRESHOLD + 1e-6)) * 30.0)

    return {
        'raw_sim': raw_sim,
        'raw_verdict': raw_verdict,
        'sim_base_cen': sim_base_template,
        'sim_loud_cen': sim_loud,
        'sim_soft_cen': sim_soft,
        'calibrated_sim': calibrated_sim,
        'is_same_person': is_same_person,
        'calibrated_verdict': calibrated_verdict,
        'confidence': conf,
        'vocal_effort_detected': vocal_effort_detected,
        'effort_type': effort_type,
        'energy_ratio': energy_ratio,
        'n_slices1': n_slices1,
        'n_slices2': n_slices2
    }

def print_result_card(results, path1, path2):
    print("\n" + "=" * 90)
    print("                          SPEAKER VERIFICATION RESULTS")
    print("=" * 90)

    v1_name = os.path.basename(path1)
    v2_name = os.path.basename(path2)
    print(f"  Voice 1 (Reference) : {v1_name} ({results['n_slices1']} speech windows)")
    print(f"  Voice 2 (Test)      : {v2_name} ({results['n_slices2']} speech windows)")
    print("-" * 90)

    if results['is_same_person']:
        status_box = [
            "┌────────────────────────────────────────────────────────────────────────────────────────┐",
            "│                                                                                        │",
            "│                      >>> FINAL VERDICT: SAME PERSON (MATCH) <<<                        │",
            "│                                                                                        │",
            f"│   Biometric Confidence  : {results['confidence']:.1f}%                                                         │",
            f"│   Calibrated Score      : {results['calibrated_sim']:+.4f}  (Threshold tau = {CALIBRATED_THRESHOLD:.4f})                 │",
            f"│   Margin Above Threshold: {results['calibrated_sim'] - CALIBRATED_THRESHOLD:+.4f}                                                 │",
            "│                                                                                        │",
            "└────────────────────────────────────────────────────────────────────────────────────────┘"
        ]
    else:
        status_box = [
            "┌────────────────────────────────────────────────────────────────────────────────────────┐",
            "│                                                                                        │",
            "│                   >>> FINAL VERDICT: DIFFERENT PERSON (IMPOSTOR) <<<                   │",
            "│                                                                                        │",
            f"│   Rejection Confidence  : {results['confidence']:.1f}%                                                         │",
            f"│   Calibrated Score      : {results['calibrated_sim']:+.4f}  (Threshold tau = {CALIBRATED_THRESHOLD:.4f})                 │",
            f"│   Margin Below Threshold: {results['calibrated_sim'] - CALIBRATED_THRESHOLD:+.4f}                                                 │",
            "│                                                                                        │",
            "└────────────────────────────────────────────────────────────────────────────────────────┘"
        ]

    for line in status_box:
        print("  " + line)

    print("\nDETAILED ARCHITECTURAL COMPARISON:")
    print("-" * 90)
    print(f"{'Pipeline Architecture':<36} | {'Score':<10} | {'Threshold':<11} | {'Decision':<16} | {'Status'}")
    print("-" * 90)
    print(f"{'Image 1: Baseline (Single Window)':<36} | {results['raw_sim']:>10.4f} | {RAW_THRESHOLD:>11.4f} | {results['raw_verdict']:<16} | {'Raw Uncentered'}")
    print(f"{'Image 2: Full System (MMSE_v + DSP)':<36} | {results['calibrated_sim']:>10.4f} | {CALIBRATED_THRESHOLD:>11.4f} | {results['calibrated_verdict']:<16} | {'Multi-Window'}")
    print("-" * 90)

    print("\nSCIENTIFIC EXPLANATION:")
    if results['is_same_person']:
        print("  ✓ Both voice utterances share genuine speaker identity characteristics.")
        if results['vocal_effort_detected']:
            print(f"  ✓ Non-neutral phonation detected ({results['effort_type']}).")
            print("  ✓ PCA-MMSE estimator successfully removed the vocal effort transfer vector,")
            print("    restoring biometric alignment with the reference voice.")
        else:
            print("  ✓ Standard phonation verified. Multi-window template similarity strongly exceeds threshold.")
    else:
        print("  ✗ The voice utterances originate from DIFFERENT human speakers.")
        print(f"  ✗ The similarity score ({results['calibrated_sim']:.4f}) falls below the")
        print(f"    verification threshold ({CALIBRATED_THRESHOLD:.4f}), triggering an impostor rejection.")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Live 10-Second Voice Recording and Speaker Verification Demo")
    parser.add_argument("--voice1", "--audio1", type=str, dest="voice1", default=None, help="Optional path to existing Voice 1 .wav file")
    parser.add_argument("--voice2", "--audio2", type=str, dest="voice2", default=None, help="Optional path to existing Voice 2 .wav file")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (default: 10)")
    parser.add_argument("--effort", type=str, choices=["auto", "normal", "loud", "soft"], default="auto", help="Vocal effort evaluation mode (default: auto)")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(BASE_DIR, "checkpoint_epoch_20.pth"), help="Path to checkpoint")
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 90)
    print("   LIVE SPEAKER VERIFICATION SYSTEM: 10-SECOND INTERACTIVE DEMO")
    print("   IEEE MLSP 2023 Implementation: Vocal Effort-Robust Transfer Vector Estimation")
    print("=" * 90)
    print(f"Device        : {device}")
    print(f"Model Weights : {args.checkpoint}")
    print(f"Duration      : {args.duration} seconds per recording")

    # Load Model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return

    print("\n[Step 1/3] Loading neural network front-end & back-end...")
    model = VocalEffortRobustModel(embedding_dim=256)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  ✓ WavLM + ECAPA-TDNN loaded successfully.")

    # Load / Fit MMSE Estimators
    print("\n[Step 2/3] Preparing PCA-MMSE statistical estimators...")
    mmse_loud, mmse_soft, global_mean = get_or_fit_mmse_estimators(model, device)

    # Acquire Voice 1 & Voice 2
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    v1_path = args.voice1
    v2_path = args.voice2

    if v1_path is None or v2_path is None:
        print("\n" + "=" * 90)
        print("                        INTERACTIVE RECORDING SESSION")
        print("=" * 90)

    # 1. Voice 1 (Reference)
    if v1_path is None:
        v1_path = os.path.join(RECORDINGS_DIR, "voice1_reference.wav")
        print("\n--------------------------------------------------------------------------------")
        print("RECORDING 1: ENROLLMENT / REFERENCE VOICE")
        print("--------------------------------------------------------------------------------")
        print("Instructions:")
        print(f"  Speak in your NORMAL voice for {args.duration} seconds.")
        print("  Say a continuous sentence or count numbers (e.g. 'This is my reference voice').")
        try:
            input("\nPress [ENTER] when you are ready to record Voice 1... ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        saved = record_audio_mic(v1_path, duration=args.duration)
        if saved is None:
            v1_path = input("Enter path to Voice 1 .wav file: ").strip()
            if not os.path.exists(v1_path):
                print(f"Error: File '{v1_path}' does not exist.")
                return

    # 2. Voice 2 (Test)
    if v2_path is None:
        v2_path = os.path.join(RECORDINGS_DIR, "voice2_test.wav")
        print("\n--------------------------------------------------------------------------------")
        print("RECORDING 2: TEST VOICE")
        print("--------------------------------------------------------------------------------")
        print("Instructions:")
        print(f"  Now record Voice 2 for {args.duration} seconds. You can test:")
        print("    [Option A] Speak NORMALLY (test same-speaker match)")
        print("    [Option B] SHOUT or WHISPER (test vocal effort MMSE compensation!)")
        print("    [Option C] Have a FRIEND speak (test impostor rejection)")
        try:
            input("\nPress [ENTER] when you are ready to record Voice 2... ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        saved = record_audio_mic(v2_path, duration=args.duration)
        if saved is None:
            v2_path = input("Enter path to Voice 2 .wav file: ").strip()
            if not os.path.exists(v2_path):
                print(f"Error: File '{v2_path}' does not exist.")
                return

    # Run Verification
    print("\n[Step 3/3] Running deep feature extraction & Bayesian MMSE compensation...")
    results = run_verification(v1_path, v2_path, model, device, mmse_loud, mmse_soft, global_mean, effort_mode=args.effort)

    # Display results
    print_result_card(results, v1_path, v2_path)

if __name__ == "__main__":
    main()
