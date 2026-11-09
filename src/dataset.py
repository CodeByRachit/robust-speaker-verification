import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AVIDTripletDataset(Dataset):
    """
    PyTorch Dataset specifically designed for the AVID (Aalto Vocal Intensity Database).
    """
    def __init__(self, root_dir, max_length_sec=4.0, sample_rate=16000):
        super(AVIDTripletDataset, self).__init__()
        
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)
        
        # Recursively gather all .wav files across Repository 1 & Repository 2,
        # excluding the Calibration_tones directory.
        raw_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        self.all_files = [f for f in raw_files if "Calibration_tones" not in f]
        
        # Structure: speaker_data[speaker_id][intensity_type] = [list_of_file_paths]
        self.speaker_data = {}
        for f in self.all_files:
            basename = os.path.basename(f)
            parts = basename.replace('.wav', '').split('_')
            
            if len(parts) >= 4:
                spk = parts[0]          # e.g., 'sp10'
                intensity = parts[-1]   # Use parts[-1] in case prefix has underscores
                
                if spk not in self.speaker_data:
                    self.speaker_data[spk] = {'normal': [], 'soft': [], 'loud': [], 'veryloud': []}
                    
                if intensity in self.speaker_data[spk]:
                    self.speaker_data[spk][intensity].append(f)
                    
        # Filter valid speakers
        self.valid_speakers = []
        for spk, data in self.speaker_data.items():
            has_normal = len(data['normal']) > 0
            has_variant = len(data['soft']) > 0 or len(data['loud']) > 0 or len(data['veryloud']) > 0
            
            if has_normal and has_variant:
                self.valid_speakers.append(spk)
                
        # Build anchor list: every normal file is included multiple times per epoch,
        # and the positive variant is dynamically sampled inside __getitem__ so 100% of files are used.
        self.anchor_files = []
        for spk in self.valid_speakers:
            for f in self.speaker_data[spk]['normal']:
                # Repeat 3x per epoch so each normal file is tested against soft, loud, and veryloud
                for _ in range(3):
                    self.anchor_files.append((spk, f))
                
    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        return waveform.squeeze(0)

    def __len__(self):
        return len(self.anchor_files)

    def __getitem__(self, idx):
        spk, anchor_path = self.anchor_files[idx]
        
        # 1. LOAD ANCHOR: Normal voice
        anchor_wav = self.load_audio(anchor_path)
        
        # 2. LOAD POSITIVE: Dynamically pick a variant file every epoch (no files left out)
        variants = (
            self.speaker_data[spk]['soft'] +
            self.speaker_data[spk]['loud'] +
            self.speaker_data[spk]['veryloud']
        )
        pos_path = random.choice(variants)
        pos_wav = self.load_audio(pos_path)
        
        # 3. LOAD NEGATIVE: Different speaker
        neg_spk = random.choice([s for s in self.valid_speakers if s != spk])
        all_neg_files = []
        for int_type in ['normal', 'soft', 'loud', 'veryloud']:
            all_neg_files.extend(self.speaker_data[neg_spk][int_type])
            
        neg_path = random.choice(all_neg_files)
        neg_wav = self.load_audio(neg_path)
        
        return anchor_wav, pos_wav, neg_wav

def get_dataloader(root_dir, batch_size=16, num_workers=0):
    dataset = AVIDTripletDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return dataloader