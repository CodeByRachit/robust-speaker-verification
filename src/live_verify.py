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

    # Calculate intra-speaker self-similarity across slices
    if len(embs) > 1:
        self_sims = [cosine_similarity(embs[i], embs[j]) for i in range(len(embs)) for j in range(i+1, len(embs))]
        self_sim = float(np.mean(self_sims))
    else:
        self_sim = 1.0

    return weighted_template, best_raw_emb, len(slices), self_sim

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
                t_n, _, _, _ = extract_template(w_n, model, device)
                t_v, _, _, _ = extract_template(w_v, model, device)
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
    Extracts embeddings for both voice recordings, runs vocal effort identification,
    applies Bayesian MMSE compensation, and evaluates biometric identity.
    """
    w1 = preprocess_audio(audio_path1)
    w2 = preprocess_audio(audio_path2)

    emb1_template, emb1_raw, n_slices1, self_sim1 = extract_template(w1, model, device)
    emb2_template, emb2_raw, n_slices2, self_sim2 = extract_template(w2, model, device)

    # Use the sharp, peak speech embedding to evaluate identity without background room-channel contamination
    emb1_peak = emb1_raw
    emb2_peak = emb2_raw

    # 1. Image 1: Raw Baseline Cosine Similarity (single-window uncompensated)
    raw_sim = cosine_similarity(emb1_peak, emb2_peak)
    raw_verdict = "SAME PERSON" if raw_sim >= RAW_THRESHOLD else "DIFFERENT PERSON"

    # 2. Image 2: Centralized Cosine Similarity in PCA-MMSE Space (Equation 10, IEEE MLSP 2023)
    sim_base_cen = centered_cosine(emb1_peak, emb2_peak, global_mean)

    # MMSE Vocal Effort Compensations in Centralized Space
    emb2_loud_comp = mmse_loud.compensate(emb2_peak)
    sim_loud_cen = centered_cosine(emb1_peak, emb2_loud_comp, global_mean)

    emb2_soft_comp = mmse_soft.compensate(emb2_peak)
    sim_soft_cen = centered_cosine(emb1_peak, emb2_soft_comp, global_mean)

    # 3. Acoustic Vocal Effort Detection from active speech dynamics (post-VAD)
    w1_raw, _ = torchaudio.load(audio_path1)
    w2_raw, _ = torchaudio.load(audio_path2)
    t1_speech = trim_silence(w1_raw)
    t2_speech = trim_silence(w2_raw)
    rms1 = float(torch.sqrt(torch.mean(t1_speech ** 2))) + 1e-8
    rms2 = float(torch.sqrt(torch.mean(t2_speech ** 2))) + 1e-8
    energy_ratio = rms2 / rms1
    delta_db = float(20.0 * np.log10(energy_ratio))

    # Vocal Effort Classification and Transfer Vector Compensation:
    if effort_mode == "loud" or (effort_mode == "auto" and delta_db >= 3.5):
        effort_class = "LOUD / SHOUTED"
        effort_desc = f"{delta_db:+.1f} dB ({energy_ratio:.1f}x relative intensity)"
        effort_acoustic = "Elevated subglottal pressure and high-frequency formant shift"
        compensation_applied = "PCA-MMSE Loud Transfer Vector Subtraction"
        calibrated_sim = sim_loud_cen
    elif effort_mode == "soft" or (effort_mode == "auto" and delta_db <= -3.5):
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
        'raw_threshold': RAW_THRESHOLD,
        'sim_base_cen': sim_base_cen,
        'sim_loud_cen': sim_loud_cen,
        'sim_soft_cen': sim_soft_cen,
        'calibrated_sim': calibrated_sim,
        'threshold': CALIBRATED_THRESHOLD,
        'self_sim1': self_sim1,
        'is_same_person': is_same_person,
        'calibrated_verdict': calibrated_verdict,
        'confidence': conf,
        'effort_class': effort_class,
        'effort_desc': effort_desc,
        'effort_acoustic': effort_acoustic,
        'compensation_applied': compensation_applied,
        'energy_ratio': energy_ratio,
        'delta_db': delta_db,
        'n_slices1': n_slices1,
        'n_slices2': n_slices2
    }

def print_result_card(results, path1, path2):
    print("\n" + "=" * 90)
    print("                     BIOMETRIC & PHONATION ANALYSIS RESULTS")
    print("=" * 90)

    v1_name = os.path.basename(path1)
    v2_name = os.path.basename(path2)
    threshold = results['threshold']
    print(f"  Voice 1 (Reference) : {v1_name} ({results['n_slices1']} speech windows, consistency: {results['self_sim1']:.4f})")
    print(f"  Voice 2 (Test)      : {v2_name} ({results['n_slices2']} speech windows)")
    print("-" * 90)

    # SECTION 1: VOCAL EFFORT IDENTIFICATION CARD
    print("\n  [1] DETECTED VOCAL EFFORT / PHONATION MODE:")
    effort_box = [
        "┌────────────────────────────────────────────────────────────────────────────────────────┐",
        "│                                                                                        │",
        f"│   Phonation State     : {results['effort_class']:<55} │",
        f"│   Acoustic Sound Level: {results['effort_desc']:<55} │",
        f"│   Spectral Dynamics   : {results['effort_acoustic']:<55} │",
        f"│   Bayesian MMSE Action: {results['compensation_applied']:<55} │",
        "│                                                                                        │",
        "└────────────────────────────────────────────────────────────────────────────────────────┘"
    ]
    for line in effort_box:
        print("  " + line)

    # SECTION 2: BIOMETRIC IDENTITY VERDICT CARD
    print("\n  [2] BIOMETRIC SPEAKER VERIFICATION VERDICT:")
    if results['is_same_person']:
        status_box = [
            "┌────────────────────────────────────────────────────────────────────────────────────────┐",
            "│                                                                                        │",
            "│                      >>> FINAL VERDICT: SAME PERSON (MATCH) <<<                        │",
            "│                                                                                        │",
            f"│   Biometric Confidence  : {results['confidence']:.1f}%                                                         │",
            f"│   Calibrated Score      : {results['calibrated_sim']:+.4f}  (Threshold tau = {threshold:.4f})                 │",
            f"│   Margin Above Threshold: {results['calibrated_sim'] - threshold:+.4f}                                                 │",
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
            f"│   Calibrated Score      : {results['calibrated_sim']:+.4f}  (Threshold tau = {threshold:.4f})                 │",
            f"│   Margin Below Threshold: {results['calibrated_sim'] - threshold:+.4f}                                                 │",
            "│                                                                                        │",
            "└────────────────────────────────────────────────────────────────────────────────────────┘"
        ]

    for line in status_box:
        print("  " + line)

    print("\nDETAILED ARCHITECTURAL COMPARISON:")
    print("-" * 90)
    print(f"{'Pipeline Architecture':<36} | {'Score':<10} | {'Threshold':<11} | {'Decision':<16} | {'Status'}")
    print("-" * 90)
    print(f"{'Image 1: Baseline (E-T+WavLM)':<36} | {results['raw_sim']:>10.4f} | {results['raw_threshold']:>11.4f} | {results['raw_verdict']:<16} | {'Raw Uncentered'}")
    print(f"{'Image 2: Full System (MMSE_v)':<36} | {results['calibrated_sim']:>+10.4f} | {threshold:>11.4f} | {results['calibrated_verdict']:<16} | {'Calibrated Centered'}")
    print("-" * 90)

    print("\nSCIENTIFIC EXPLANATION:")
    print(f"  ✓ Phonation Analysis: Test voice was classified as {results['effort_class']}.")
    if "LOUD" in results['effort_class'] or "SOFT" in results['effort_class']:
        print("  ✓ Non-neutral phonation detected. The PCA-MMSE estimator subtracted the estimated")
        print("    transfer vector to eliminate vocal effort distortion from speaker identity.")
    else:
        print("  ✓ Conversational speech verified. Standard neutral speech scoring applied.")

    if results['is_same_person']:
        print("  ✓ Biometric Decision: Both voice utterances share genuine speaker identity characteristics.")
        print(f"  ✓ Calibrated similarity score ({results['calibrated_sim']:+.4f}) exceeds the verification threshold ({threshold:.4f}).")
    else:
        print("  ✗ Biometric Decision: The voice utterances originate from DIFFERENT human speakers.")
        print(f"  ✗ Calibrated similarity score ({results['calibrated_sim']:+.4f}) falls below verification threshold ({threshold:.4f}), triggering rejection.")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Live Interactive Voice Recording and Speaker Verification Demo")
    parser.add_argument("--voice1", "--audio1", type=str, dest="voice1", default=None, help="Optional path to existing Voice 1 .wav file")
    parser.add_argument("--voice2", "--audio2", type=str, dest="voice2", default=None, help="Optional path to existing Voice 2 .wav file")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds (default: 5)")
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
        print("    [Option A] Speak NORMALLY (test same-speaker match under normal effort)")
        print("    [Option B] SHOUT or speak LOUDLY (test loud effort detection & MMSE compensation)")
        print("    [Option C] WHISPER or speak SOFTLY (test soft effort detection & MMSE compensation)")
        print("    [Option D] Have a FRIEND speak (test impostor rejection)")
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
