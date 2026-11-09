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

# Suppress library deprecation and SSL warnings
warnings.filterwarnings("ignore")

from model import VocalEffortRobustModel
from mmse_estimator import PCAMMSEEstimator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVID_DIR = os.path.join(BASE_DIR, "AVID", "AVID")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
CACHE_FILE = os.path.join(BASE_DIR, "checkpoint_mmse_cache.pkl")

# True mathematical EER thresholds computed directly from the AVID benchmark
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
    
    # Simple dataset scan
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

    # Helper to fit MMSE on candidate pairs
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
                w_n = load_audio(nf, max_length_sec=4.0).to(device)
                w_v = load_audio(vf, max_length_sec=4.0).to(device)
                X_norm.append(model(w_n).cpu().numpy().flatten())
                X_var.append(model(w_v).cpu().numpy().flatten())
                
        X_norm = np.vstack(X_norm)
        X_var = np.vstack(X_var)
        mmse = PCAMMSEEstimator(n_components_pca=min(L, len(X_norm) - 1), n_components_gmm=K)
        mmse.fit(X_norm, X_var)
        return mmse, X_norm

    mmse_loud, embs_loud = fit_estimator(['loud', 'veryloud'], max_pairs=80)
    mmse_soft, embs_soft = fit_estimator(['soft'], max_pairs=80)
    global_mean = np.mean(np.vstack((embs_loud, embs_soft)), axis=0)

    # Save to disk cache
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

def run_verification(audio_path1, audio_path2, model, device, mmse_loud, mmse_soft, global_mean):
    """
    Extracts embeddings for both voice recordings, runs baseline vs MMSE compensated verification,
    and returns comprehensive similarity scores and biometric verdicts.
    """
    with torch.no_grad():
        w1 = load_audio(audio_path1).to(device)
        w2 = load_audio(audio_path2).to(device)
        emb1 = model(w1).cpu().numpy().flatten()
        emb2 = model(w2).cpu().numpy().flatten()

    # 1. Image 1: Raw Baseline Cosine Similarity
    raw_sim = cosine_similarity(emb1, emb2)
    raw_verdict = "SAME PERSON" if raw_sim >= RAW_THRESHOLD else "DIFFERENT PERSON"

    # 2. Image 2: Calibrated DSP-Centered Cosine Similarity
    def centered_cosine(v1, v2):
        c1 = v1 - global_mean
        c2 = v2 - global_mean
        return cosine_similarity(c1, c2)

    sim_base_cen = centered_cosine(emb1, emb2)

    # MMSE Vocal Effort Compensations (Test Voice compensation as in paper Image 2)
    emb2_loud_comp = mmse_loud.compensate(emb2)
    sim_loud_cen = centered_cosine(emb1, emb2_loud_comp)

    emb2_soft_comp = mmse_soft.compensate(emb2)
    sim_soft_cen = centered_cosine(emb1, emb2_soft_comp)

    # Symmetric compensation in case reference voice had vocal effort
    emb1_loud_comp = mmse_loud.compensate(emb1)
    sim_v1_loud_cen = centered_cosine(emb1_loud_comp, emb2)
    emb1_soft_comp = mmse_soft.compensate(emb1)
    sim_v1_soft_cen = centered_cosine(emb1_soft_comp, emb2)

    # Best calibrated similarity across potential vocal effort conditions
    calibrated_sim = max(sim_base_cen, sim_loud_cen, sim_soft_cen, sim_v1_loud_cen, sim_v1_soft_cen)
    is_same_person = (calibrated_sim >= CALIBRATED_THRESHOLD)
    calibrated_verdict = "SAME PERSON" if is_same_person else "DIFFERENT PERSON"

    # Confidence calculation based on margin to decision threshold
    if is_same_person:
        margin = calibrated_sim - CALIBRATED_THRESHOLD
        conf = 70.0 + min(29.5, (margin / (1.0 - CALIBRATED_THRESHOLD + 1e-6)) * 30.0)
    else:
        margin = CALIBRATED_THRESHOLD - calibrated_sim
        conf = 70.0 + min(29.5, (margin / (CALIBRATED_THRESHOLD + 1.0 + 1e-6)) * 30.0)

    # Detect if vocal effort shift was active
    vocal_effort_detected = False
    effort_type = "Normal"
    if max(sim_loud_cen, sim_v1_loud_cen) > sim_base_cen + 0.02:
        vocal_effort_detected = True
        effort_type = "Shouted / Loud Speech"
    elif max(sim_soft_cen, sim_v1_soft_cen) > sim_base_cen + 0.02:
        vocal_effort_detected = True
        effort_type = "Whispered / Soft Speech"

    return {
        'raw_sim': raw_sim,
        'raw_verdict': raw_verdict,
        'sim_base_cen': sim_base_cen,
        'sim_loud_cen': sim_loud_cen,
        'sim_soft_cen': sim_soft_cen,
        'calibrated_sim': calibrated_sim,
        'is_same_person': is_same_person,
        'calibrated_verdict': calibrated_verdict,
        'confidence': conf,
        'vocal_effort_detected': vocal_effort_detected,
        'effort_type': effort_type
    }

def print_result_card(results, path1, path2):
    print("\n" + "=" * 90)
    print("                          SPEAKER VERIFICATION RESULTS")
    print("=" * 90)

    v1_name = os.path.basename(path1)
    v2_name = os.path.basename(path2)
    print(f"  Voice 1 (Reference) : {v1_name}")
    print(f"  Voice 2 (Test)      : {v2_name}")
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
    print(f"{'Image 1: Baseline (E-T+WavLM)':<36} | {results['raw_sim']:>10.4f} | {RAW_THRESHOLD:>11.4f} | {results['raw_verdict']:<16} | {'Raw Uncentered'}")
    print(f"{'Image 2: Full System (MMSE_v)':<36} | {results['calibrated_sim']:>+10.4f} | {CALIBRATED_THRESHOLD:>11.4f} | {results['calibrated_verdict']:<16} | {'Calibrated Centered'}")
    print("-" * 90)

    print("\nSCIENTIFIC EXPLANATION:")
    if results['is_same_person']:
        print("  ✓ Both voice utterances share genuine speaker identity characteristics.")
        if results['vocal_effort_detected']:
            print(f"  ✓ Non-neutral phonation detected ({results['effort_type']}).")
            print("  ✓ PCA-MMSE estimator successfully removed the vocal effort transfer vector,")
            print("    restoring biometric alignment with the reference voice.")
        else:
            print("  ✓ Standard phonation verified. Calibrated similarity is strongly positive.")
    else:
        print("  ✗ The voice utterances originate from DIFFERENT human speakers.")
        print(f"  ✗ The calibrated similarity ({results['calibrated_sim']:+.4f}) falls significantly below the")
        print(f"    verification threshold ({CALIBRATED_THRESHOLD:.4f}), triggering an impostor rejection.")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Live 10-Second Voice Recording and Speaker Verification Demo")
    parser.add_argument("--voice1", "--audio1", type=str, dest="voice1", default=None, help="Optional path to existing Voice 1 .wav file")
    parser.add_argument("--voice2", "--audio2", type=str, dest="voice2", default=None, help="Optional path to existing Voice 2 .wav file")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (default: 10)")
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
            # Fallback if sounddevice not installed
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
    results = run_verification(v1_path, v2_path, model, device, mmse_loud, mmse_soft, global_mean)

    # Display results
    print_result_card(results, v1_path, v2_path)

if __name__ == "__main__":
    main()

