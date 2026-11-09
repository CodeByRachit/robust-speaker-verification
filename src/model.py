import torch
import torch.nn as nn
import torchaudio
from transformers import WavLMModel
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class VocalEffortRobustModel(nn.Module):
    """
    Vocal Effort-Robust Speaker Verification Model.
    
    Front-End: WavLM (self-supervised model for speech representation).
    Back-End: ECAPA-TDNN (with attentive statistics pooling).
    """
    def __init__(self, wavlm_path="microsoft/wavlm-base-plus", ecapa_channels=[512, 512, 512, 512, 1536], embedding_dim=256):
        super(VocalEffortRobustModel, self).__init__()
        
        # 1. Front-End: WavLM
        # We load a pre-trained WavLM model. 
        # For speaker verification, we often freeze it or fine-tune it carefully.
        self.wavlm = WavLMModel.from_pretrained(wavlm_path)
        
        # 2. Back-End: ECAPA-TDNN
        # WavLM-base-plus outputs hidden states of dimension 768.
        # We pass these through ECAPA-TDNN to get speaker embeddings.
        wavlm_hidden_size = self.wavlm.config.hidden_size
        
        self.ecapa_tdnn = ECAPA_TDNN(
            input_size=wavlm_hidden_size,
            channels=ecapa_channels,
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=embedding_dim
        )
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x (torch.Tensor): Raw waveform tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Mask for padded inputs.
        Returns:
            torch.Tensor: Speaker embeddings of shape (batch_size, embedding_dim).
        """
        # Pass through WavLM
        # WavLM expects 16kHz audio.
        wavlm_out = self.wavlm(x, attention_mask=attention_mask)
        
        # We take the last hidden state from WavLM
        # Shape: (batch_size, sequence_length, hidden_size)
        hidden_states = wavlm_out.last_hidden_state
        
        # Pass through ECAPA-TDNN
        # SpeechBrain's ECAPA_TDNN expects input of shape (batch_size, time, channels)
        embeddings = self.ecapa_tdnn(hidden_states)
        
        # The output of ECAPA-TDNN might have an extra dimension depending on the version
        # Usually shape is (batch_size, 1, embedding_dim)
        if len(embeddings.shape) == 3:
            embeddings = embeddings.squeeze(1)
            
        return embeddings

def compute_cosine_similarity(emb1, emb2):
    """
    Computes cosine similarity between two embeddings.
    """
    cos = nn.CosineSimilarity(dim=-1)
    return cos(emb1, emb2)
