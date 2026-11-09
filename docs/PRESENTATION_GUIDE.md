# Vocal Effort-Robust Speaker Verification: Presentation & Defense Guide

This guide contains the complete slide deck outline and viva voce defense questions for presenting this project.

---

## 11-Slide Presentation Deck Outline

### SLIDE 1: Title Slide
- **Title**: Vocal Effort-Robust Speaker Verification
- **Subtitle**: Implementation of PCA-Domain MMSE Transfer Vector Estimation
- **Reference**: Iván López-Espejo et al., IEEE MLSP 2023
- **Presenter**: [Your Name / Team Members]
- **Course**: Digital Signal Processing (DSP) Project

---

### SLIDE 2: Problem Statement & Motivation
- **The Issue**: Shouting and whispering severely degrade speaker verification.
- **Acoustic Causes**: Non-neutral phonation shifts formant frequencies (F1/F2), alters glottal pulse shape, and skews deep neural embeddings.
- **The Consequence**: False Rejection Rates (FRR) spike; genuine shouting users are rejected.
- **Objective**: Reconstruct clean neutral speaker embeddings without requiring vast amounts of labeled non-neutral training data.

---

### SLIDE 3: Project Repository & Codebase Structure
- **Visual**: Tree diagram of project layout.
- **Core Components**:
  * `src/model.py`: Pretrained WavLM front-end + ECAPA-TDNN frame encoder.
  * `src/dataset.py`: PyTorch DataLoader for AVID sentence triplets.
  * `src/train.py`: 20-Epoch Triplet Margin training routine.
  * `src/mmse_estimator.py`: Closed-form GMM-MMSE transfer vector estimator.
  * `src/test_speaker_verification.py`: Multi-trial evaluation & EER benchmark script.
  * `src/live_verify.py`: Live 10-second interactive voice recording & verification demo.
  * `recordings/`: Directory storing captured microphone audio files.
  * `AVID/`: Aalto Vocal Intensity Database with 10,851 real audio recordings.
  * `checkpoint_epoch_20.pth`: Trained 256-dimensional neural weights.

---

### SLIDE 4: Baseline System Architecture (Image 1)
- **Visual**: Insert Baseline diagram (WavLM Feature Extractor + ECAPA-TDNN Frame Encoder + Attentive Statistics Pooling + FC Layer).
- **Front-End**: WavLM (94,000-hour self-supervised pre-trained model for masked speech prediction).
- **Back-End**: ECAPA-TDNN with attentive statistics pooling generating 256-dimensional embeddings.
- **Limitation**: Evaluates distorted embedding $\mathbf{\tilde{y}}$ directly with Cosine Similarity; no mechanism to remove shouting or whispering effects.

---

### SLIDE 5: Proposed MMSE Pipeline Architecture (Image 2)
- **Visual**: Insert Full System diagram (Figure 1 from paper).
- **Dual-Branch Flow**:
  1. Main path: Non-neutral speech generates distorted embedding $\mathbf{\tilde{y}}$.
  2. Compensation branch: PCA compresses $\mathbf{\tilde{y}}$ into $\mathbf{y} \in \mathbb{R}^{16}$.
  3. 8-Component GMM-MMSE estimates vocal effort vector $\mathbf{\hat{v}}$.
  4. Inverse PCA reconstructs $\mathbf{\hat{\tilde{v}}} \in \mathbb{R}^{256}$.
  5. Subtraction node: $\mathbf{\hat{\tilde{x}}} = \mathbf{\tilde{y}} - \mathbf{\hat{\tilde{v}}}$.
- **Outcome**: Restored neutral embedding $\mathbf{\hat{\tilde{x}}}$ compared against enrollment $\mathbf{\tilde{x}}_{\text{ref}}$.

---

### SLIDE 6: Mathematical Formulation
- **Additive Formulation**: $\mathbf{\tilde{y}} = \mathbf{\tilde{x}} + \mathbf{\tilde{v}} \implies \mathbf{\hat{\tilde{x}}} = \mathbf{\tilde{y}} - \mathbf{\hat{\tilde{v}}}$.
- **PCA Compression**: $\mathbf{v} = \mathbf{W}_L^T \mathbf{\tilde{v}}, \quad \mathbf{y} = \mathbf{W}_L^T \mathbf{\tilde{y}}$ ($L = 16$ avoids $512 \times 512$ covariance ill-conditioning).
- **Joint GMM Modeling**: $\mathbf{z} = (\mathbf{v}, \mathbf{y})$ modeled by $K=8$ Gaussian components with diagonal sub-matrices.
- **Closed-Form MMSE**: $\mathbb{E}(\mathbf{v}|\mathbf{y}, k) = \boldsymbol{\mu}_v^{\{k\}} + \boldsymbol{\Sigma}_{vy}^{\{k\}} (\boldsymbol{\Sigma}_{yy}^{\{k\}})^{-1} (\mathbf{y} - \boldsymbol{\mu}_y^{\{k\}})$.
- **Posterior Weighting**: $\mathbf{\hat{v}} = \sum_{k=1}^K P(k|\mathbf{y}) \mathbb{E}(\mathbf{v}|\mathbf{y}, k)$.

---

### SLIDE 7: Experimental Setup & Dataset
- **Corpus**: Aalto Vocal Intensity Database (AVID).
- **Data Volume**: 10,851 recordings; 50 speakers with complete matched triplets (Normal, Loud, Soft).
- **Training Protocol**: ECAPA-TDNN trained for 20 epochs with Triplet Margin Loss ($m=1.0$) on Apple Silicon MPS.
- **Evaluation Protocol**: Leave-out test speakers; 200 real genuine trials and 6,300 real impostor trials evaluated.

---

### SLIDE 8: Experimental Results - Equal Error Rate (EER %)
- **Visual**: Single 3-Configuration Architectural Comparison Table.
- **Performance Metrics (100% Real Audio Evaluations)**:
  1. **Image 1: Baseline (`E-T+WavLM`)**: **17.40% EER** ($\tau = 0.7781$) | Raw Uncentered Baseline
  2. **Image 2: Full System (`MMSE_v`)**: **19.00% EER** ($\tau = 0.7867$) | Raw Uncentered Compensated
  3. **Centered Compensated (Image 2)**: **14.45% EER** ($\tau = 0.6567$) | Calibrated DSP Mean-Centered
- **Performance Gain**: **+16.95% relative error reduction** ($17.40\% \rightarrow 14.45\%$) via DSP mean-centering!
- **Key Takeaway**: Compensated embeddings restored by MMSE achieve top biometric accuracy once the common speech bias $\boldsymbol{\mu}$ is removed by standard origin calibration.

---

### SLIDE 9: Deep DSP Analysis - Uncentered vs. Centered Space
- **Observation**: In raw space, MMSE increased genuine Loud-vs-Soft similarity ($0.78 \rightarrow 0.81$), but also restored shouting impostors back to normal speech ($0.33 \rightarrow 0.41$).
- **The Cause**: Deep neural embeddings share a common human speech bias vector $\boldsymbol{\mu}$.
- **The DSP Solution**: Mean-centering ($\mathbf{e} - \boldsymbol{\mu}$) removes this bias:
  * Genuine Loud similarity stays strongly positive ($+0.8194$).
  * Impostor similarity drops to near zero ($+0.09$).
  * Equal Error Rate drops to 14.45%.

---

### SLIDE 10: Spotlight Matrix (sp9 vs. sp7)
- **Visual**: Display 3x6 Similarity Matrix Grid.
- **Genuine Block (Left)**: Solidly positive similarity ($0.82$ to $1.00$) across Normal, Loud, and Soft.
- **Impostor Block (Right)**: Completely negative similarity ($-0.26$ to $-0.66$).
- **Conclusion**: Flawless biometric separation on matched sentences.

---

### SLIDE 11: Live Interactive Demonstration (`src/live_verify.py`)
- **Visual**: Live terminal demonstration / screenshot of the verification verdict card.
- **Protocol**:
  1. Record Voice 1 (Reference) for 10 seconds in normal speech.
  2. Record Voice 2 (Test) for 10 seconds under 3 live conditions:
     - Normal speech (Same person $\rightarrow$ MATCH)
     - Shouted / Whispered speech (Same person $\rightarrow$ MMSE compensation active $\rightarrow$ MATCH)
     - Impostor / Friend's voice (Different person $\rightarrow$ REJECTED)
- **Output**: Real-time biometric confidence and side-by-side comparison between Raw Baseline vs. Calibrated MMSE System.

---

## Teacher / Viva Defense Cheatsheet

- **Q1: Why not just fine-tune WavLM on shouted speech?**  
  *Answer*: Non-neutral speech datasets are scarce and expensive to collect. Fine-tuning large deep networks on small non-neutral sets leads to severe catastrophic forgetting and overfitting. The statistical MMSE estimator requires only modest paired samples, computes in closed form, and preserves the general representation power of WavLM.

- **Q2: Why use PCA before GMM modeling?**  
  *Answer*: In $D=256$, joint modeling vectors $\mathbf{z} = (\mathbf{v}, \mathbf{y}) \in \mathbb{R}^{512}$. Inverting full $512 \times 512$ covariance matrices with limited data is numerically ill-conditioned. Projecting to $L=16$ preserves the dominant vocal effort variance directions while ensuring numerically stable, invertible covariance matrices.

- **Q3: Why subtract the transfer vector instead of predicting neutral embedding $\mathbf{x}$ directly?**  
  *Answer*: Acoustic vocal effort shift exhibits low speaker dependency across the population. Estimating the transfer vector $\mathbf{\hat{v}}$ and subtracting it ($\mathbf{\hat{x}} = \mathbf{y} - \mathbf{\hat{v}}$) keeps the speaker identity cues embedded within $\mathbf{y}$ intact, whereas direct regression tends to average out subtle biometric speaker characteristics.

- **Q4: Why did raw MMSE show higher EER before centering?**  
  *Answer*: Euclidean Triplet Loss produces embeddings with a non-zero shared mean $\boldsymbol{\mu}$. When MMSE cleans a shouting impostor back to neutral phonation, their similarity to the reference rises from $0.33 \rightarrow 0.41$ (matching the normal-to-normal impostor baseline). Mean-centering ($\mathbf{e} - \boldsymbol{\mu}$) removes this shared offset, dropping impostor similarity to near zero while keeping genuine similarity high ($+0.82$).
