"""
FINGERPRINT RECTIFICATION TRAINING SYSTEM
=========================================

1. Takes altered fingerprints as input, tries to produce "fixed" originals.
2. After training, can process a folder of altered fingerprints and output fixed versions.
3. Uses encoder-decoder CNN for rectification.
"""
import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from . import stn_siamese as SIFTSN
# ============================================================
# RECTIFICATION NETWORK
# ============================================================
class FingerprintSiftMatcher(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ============================================================
# DATASET FOR RECTIFICATION
# ============================================================
class FingerprintRectificationDataset(Dataset):
    """
    Dataset returns (altered_image, original_image) pairs
    """
    def __init__(self, originals_folder, altered_folders):
        self.pairs = []
        self.images = {}
        
        # Load original images
        original_files = glob(os.path.join(originals_folder, "*.BMP")) + glob(os.path.join(originals_folder, "*.bmp"))
        if not original_files:
            raise FileNotFoundError("No originals found")
        
        for f in original_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (128,128)).astype(np.float32)/255.0
            self.images[f] = img[np.newaxis, ...]  # shape: [1,H,W]

        # Load altered images
        altered_files = []
        for folder in altered_folders:
            altered_files += glob(os.path.join(folder, "*.BMP")) + glob(os.path.join(folder, "*.bmp"))

        # Match altered with original by finger_id

        for index,f in enumerate(altered_files):
            print(f"handling {index}",end="\r")
            altered_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if altered_img is None: continue
            altered_img = cv2.resize(altered_img, (128,128)).astype(np.float32)/255.0
            altered_key = self._extract_finger_id(os.path.basename(f))
            
            # find corresponding original
            orig_match = None
            for orig_path in self.images:
                if self._extract_finger_id(os.path.basename(orig_path)) == altered_key:
                    orig_match = self.images[orig_path]
                    break
            if orig_match is not None:
                self.pairs.append((altered_img[np.newaxis,...], orig_match))

    def _extract_finger_id(self, filename):
        suffixes = ['_CR', '_Obl', '_Impression', '_LowPressure', '_HighPressure', '_Dry', '_Wet']
        name = os.path.splitext(filename)[0]
        for s in suffixes:
            if name.endswith(s):
                name = name[:-len(s)]
                break
        return name

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        altered, original = self.pairs[idx]
        return torch.tensor(altered, dtype=torch.float32), torch.tensor(original, dtype=torch.float32)

# ============================================================
# TRAINER
# ============================================================
class SiftTrainer:
    def __init__(self, resources_path="../SOCOFing", output_path="./rectified_model"):
        self.resources = resources_path
        self.out_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.ckpt = os.path.join(output_path, "checkpoint.pth")

    def save_ckpt(self, model, opt, epoch, batch, loss):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "loss": loss,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, self.ckpt + ".tmp")
        os.replace(self.ckpt + ".tmp", self.ckpt)
        print(f"💾 Checkpoint saved @ epoch {epoch}, batch {batch}, loss {loss:.4f}")

    def load_ckpt(self, model, opt):
        if not os.path.exists(self.ckpt): return None
        try:
            ckpt = torch.load(self.ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"✅ Resumed from epoch {ckpt['epoch']} batch {ckpt['batch']}")
            return ckpt
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            return None

    def train(self, epochs=20, batch_size=16, save_every=500):
        originals = os.path.join(self.resources, "Real")
        altered_folders = [os.path.join(self.resources, f"Altered/{x}") for x in ["Altered-Easy","Altered-Medium","Altered-Hard"]]

        print(f"length of altered folders {len(altered_folders)}")
        dataset = FingerprintRectificationDataset(originals, altered_folders)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FingerprintSiftMatcher().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        ckpt = self.load_ckpt(model, optimizer)
        start_epoch, gbatch = (ckpt["epoch"], ckpt["batch"]) if ckpt else (0,0)

        print("start learning")
        for epoch in range(start_epoch, epochs):
            print(f" epoch : {epoch} / {epochs}")
            model.train()
            for b, (altered, original) in enumerate(loader):
                print(f"invoked {b} ",end="\r")
                altered = altered.to(device)
                original = original.to(device)

                optimizer.zero_grad()
                output = model(altered)
                loss = criterion(output, original)
                loss.backward()
                optimizer.step()

                gbatch += 1
                print(f"Epoch {epoch+1}, Batch {b+1}, Loss: {loss.item():.4f}", end="\r")
                if gbatch % save_every == 0:
                    self.save_ckpt(model, optimizer, epoch, b, loss.item())

            print(f"\nEpoch {epoch+1} completed. Loss: {loss.item():.4f}")
            self.save_ckpt(model, optimizer, epoch, b, loss.item())

        torch.save(model.state_dict(), os.path.join(self.out_path, "final_model.pth"))
        print("🎉 Training finished and final model saved!")
        return model

    # ============================================================
    # FUNCTION TO RECTIFY A FOLDER
    # ============================================================
    def rectify_folder(self, model_path, input_folder, output_folder=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FingerprintSiftMatcher().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        if output_folder is None:
            output_folder = input_folder + "_fixed"
        os.makedirs(output_folder, exist_ok=True)

        for f in os.listdir(input_folder):
            fpath = os.path.join(input_folder, f)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (128,128)).astype(np.float32)/255.0
            img_tensor = torch.tensor(img[np.newaxis, np.newaxis,...], dtype=torch.float32).to(device)

            with torch.no_grad():
                fixed = model(img_tensor).cpu().numpy()[0,0]

            # Convert back to 0-255 image
            fixed_img = (fixed*255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f), fixed_img)
            print(f"Fixed {f}")

        print(f"✅ All images saved to {output_folder}")

  

class SiftMatcherModel(SIFTSN.RectificationTrainer):
    def __init(selef,**args):
           super.__init__(args)