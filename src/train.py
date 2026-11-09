import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import VocalEffortRobustModel
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train(
    data_dir,
    num_epochs=15,
    start_epoch=5,
    resume_checkpoint=os.path.join(BASE_DIR, "checkpoint_epoch_5.pth"),
    batch_size=8,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
):
    print(f"Using device: {device}")
    
    # 1. Initialize DataLoader across all AVID repositories
    print(f"Loading dataset from {data_dir}...")
    dataloader = get_dataloader(data_dir, batch_size=batch_size)
    print(f"Total batches per epoch: {len(dataloader)}")
    
    # 2. Initialize Model
    print("Initializing model...")
    model = VocalEffortRobustModel(embedding_dim=256)
    
    # Resume from checkpoint_epoch_5.pth if it exists
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        print(f"Resumed weights from {resume_checkpoint} (continuing from Epoch {start_epoch + 1})")
    else:
        start_epoch = 0
        print("Training from scratch...")
        
    model.to(device)
    
    # Freeze WavLM to save memory and focus on training ECAPA-TDNN
    for param in model.wavlm.parameters():
        param.requires_grad = False
        
    # 3. Setup Loss and Optimizer (kept identical to epochs 1-5 for smooth continuation)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # 4. Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for anchor_wav, pos_wav, neg_wav in progress_bar:
            anchor_wav = anchor_wav.to(device)
            pos_wav = pos_wav.to(device)
            neg_wav = neg_wav.to(device)
            
            optimizer.zero_grad()
            
            anchor_emb = model(anchor_wav)
            pos_emb = model(pos_wav)
            neg_emb = model(neg_wav)
            
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(BASE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        
    print("Training complete!")

if __name__ == "__main__":
    avid_data_path = os.path.join(BASE_DIR, "AVID", "AVID")

    if os.path.exists(avid_data_path):
        # Resume from checkpoint_epoch_5.pth and train Epochs 6 to 20
        train(
            data_dir=avid_data_path,
            num_epochs=20,
            start_epoch=5,
            resume_checkpoint=os.path.join(BASE_DIR, "checkpoint_epoch_5.pth"),
            batch_size=4
        )