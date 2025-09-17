import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class TrainingModel():
    def __init__(self, **args):
        if "resources_path" in  args:
            self.resources_path = args["resources_path"]
        else :
            self.resources_path = "Images"
            
        if "find_dest_mode" in args:
            self.find_dest_mode = args['find_dest_mode']
        else:
            self.find_dest_mode = 'diff_folders_same_name'


        if "i_limit" in args:
            self.i_limit = args['i_limit']

        else:
            self.i_limit = None


        if "epochs" in args:
            self.epochs = args["epochs"] 
        else :
            self.epochs = 5   

        if "learning_results_path" in args:
            self.learning_results_path = args["learning_results_path"]
        else:
            self.learning_results_path = "./learning_results"

    def get_pretrained_model_path(self):
        return os.path.join(self.learning_results_path, "fingerprint_cnn.pth")

    def train_model(self, altered_f=None, batch_size=10):
        # default altered folders matching SOCOFing naming
        if altered_f is None:
            altered_f = ["Altered-Easy", "Altered-Medium", "Altered-Hard"]

        # Load originals into dict keyed by filename without extension
        originals_folder = os.path.join(self.resources_path, "Real")
        print("Loading originals from:", originals_folder)
        originals = load_images_from_folder(originals_folder, return_dict=True)
        if not originals:
            raise ValueError(f"No originals loaded from {originals_folder} - check your path")

        print(f"Loaded {len(originals)} original images")

        # helper to match a distorted filename base to an original key
        def find_original_key(name_no_ext, originals_dict):
            """
            Try exact match, otherwise progressively strip trailing segments
            separated by '_' until a match is found.
            Example:
             - name_no_ext = '1__M_Left_index_finger_CR'
             - will try:
                '1__M_Left_index_finger_CR' -> no
                '1__M_Left_index_finger' -> yes
            Returns the matching key or None.
            """
            if name_no_ext in originals_dict:
                return name_no_ext
            parts = name_no_ext.split("_")
            # remove one segment at a time from the end (ensures we remove distortion suffix like CR, OBI)
            for cut in range(1, len(parts)):
                candidate = "_".join(parts[:-cut])
                if candidate in originals_dict:
                    return candidate
            return None

        X_distorted = []
        Y_original = []

        # iterate the altered folders
        for fname in altered_f:
            path = os.path.join(f"./{self.resources_path}/Altered/{fname}") 
            print("Scanning altered folder:", path)
            ims = glob(os.path.join(path, "*.BMP")) + glob(os.path.join(path, "*.bmp"))  # accept both cases
            if len(ims) == 0:
                # don't fail immediately — print helpful message and continue
                print(f"Warning: found 0 images in {path} (skipping). Make sure folder exists and contains BMP files.")
                continue

            for img_path in ims:
                base = os.path.basename(img_path)
                name_no_ext = os.path.splitext(base)[0]

                orig_key = find_original_key(name_no_ext, originals)
                if orig_key is None:
                    # no matching original found -> skip with warning
                    print(f"Warning: no original found for distorted image {base} (tried '{name_no_ext}'). Skipping.")
                    continue

                # read distorted image robustly
                img_dist = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_dist is None:
                    print(f"Warning: failed to read {img_path}. Skipping.")
                    continue

                # normalize / resize and ensure shape (H,W,1)
                try:
                    img_dist = cv2.resize(img_dist, (128, 128))
                except Exception as e:
                    print(f"Warning: resize failed for {img_path}: {e}. Skipping.")
                    continue

                img_dist = img_dist.astype(np.float32) / 255.0
                img_dist = img_dist[..., np.newaxis]  # (H,W,1)
                X_distorted.append(img_dist)

                # corresponding original (already resized/normalized in load_images_from_folder)
                orig_img = originals[orig_key]
                # orig_img should already be shape (H,W,1)
                Y_original.append(orig_img)

        # Final check: at least some pairs loaded
        if len(X_distorted) == 0:
            raise ValueError("No distorted-original pairs were collected. Check folder names and filename patterns.")

        # Convert lists to arrays - ensure they are regular 4D arrays
        X_distorted = np.array(X_distorted, dtype=np.float32)
        Y_original = np.array(Y_original, dtype=np.float32)

        print("X_distorted final shape:", X_distorted.shape)
        print("Y_original final shape:", Y_original.shape)

        if X_distorted.ndim != 4 or Y_original.ndim != 4:
            raise ValueError(f"Wrong shapes: X_distorted.ndim={X_distorted.ndim}, Y_original.ndim={Y_original.ndim}. Each must be (N,H,W,1).")

        # Optionally balance or randomize pairing — currently pairs are 1:1 by construction
        # Convert to tensors (N,H,W,C) -> (N,C,H,W)
        X_tensor = torch.tensor(X_distorted, dtype=torch.float32).permute(0, 3, 1, 2)
        Y_tensor = torch.tensor(Y_original, dtype=torch.float32).permute(0, 3, 1, 2)

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Model
        model = FingerprintCNN()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load pretrained if exists
        pretrained_path = self.get_pretrained_model_path()
        if os.path.exists(pretrained_path):
            try:
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
                print(f"Successfully loaded pretrained model from {pretrained_path}")
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Starting training from scratch")
        else:
            print("No pretrained model found. Training from scratch.")

        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("start training")
        for epoch in range(self.epochs):
            print("training epoch:", epoch + 1)
            model.train()
            total_loss = 0.0
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        os.makedirs(self.learning_results_path, exist_ok=True)
        torch.save(model.state_dict(), pretrained_path)
        print("Training finished. Model saved to:", pretrained_path)

    # def match_fingerprint(self, distorted_img_path, originals_folder="Real",get_real = True):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     # Load pretrained model
    #     model = FingerprintCNN()
    #     pretrained_path = self.get_pretrained_model_path()
    #     print(pretrained_path)
    #     if not os.path.exists(pretrained_path):
    #         raise FileNotFoundError(f"No pretrained model found at {pretrained_path}")
    #     model.load_state_dict(torch.load(pretrained_path, map_location=device))
    #     model.to(device)
    #     model.eval()

    #     # Load distorted image
    #     img_gray = cv2.imread(distorted_img_path, cv2.IMREAD_GRAYSCALE)
    #     if img_gray is None:
    #         raise ValueError(f"Failed to read {distorted_img_path}")

    #     img_dist = cv2.resize(img_gray, (128, 128)).astype(np.float32) / 255.0
    #     img_dist = img_dist[np.newaxis, ..., np.newaxis]  # (1,H,W,1)
    #     tensor_dist = torch.tensor(img_dist, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    #     # Reconstruct distorted image
    #     with torch.no_grad():
    #         reconstructed = model(tensor_dist).cpu().numpy()[0, 0]  # shape (128,128)

    #     # Load all original fingerprints as array
    #     originals = load_images_from_folder(os.path.join(self.resources_path, originals_folder), return_dict=False)
    #     if len(originals) == 0:
    #         raise ValueError("No originals found for matching.")

    #     best_score = -1.0
    #     best_match = None
    #     index = None
    #     for i, orig in enumerate(originals):
    #         orig_resized = orig[..., 0]  # shape (128,128)
    #         # Compute SSIM similarity (closer to 1 is better)
    #         score = ssim(reconstructed, orig_resized, data_range=1.0)
    #         if score > best_score:
    #             best_score = score
    #             best_match = i  # index of best match
    #             data = orig
    #             index = i
    #     plt.figure(figsize=(8, 4))
    #     # Show distorted
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img_gray, cmap="gray")
    #     plt.title("Distorted")
    #     plt.axis("off")

    #     # Show original
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(data[..., 0], cmap="gray")
    #     plt.title("Original")
    #     plt.axis("off")
    #     plt.show()

    #     return best_match, best_score, index


    def match_fingerprint(self, distorted_img_path, originals_folder="Real"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model
        model = FingerprintCNN()
        pretrained_path = self.get_pretrained_model_path()
        print(pretrained_path)
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"No pretrained model found at {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.to(device)
        model.eval()

        # Load distorted image
        img_gray = cv2.imread(distorted_img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Failed to read {distorted_img_path}")

        img_dist = cv2.resize(img_gray, (128, 128)).astype(np.float32) / 255.0
        img_dist = img_dist[np.newaxis, ..., np.newaxis]  # (1,H,W,1)
        tensor_dist = torch.tensor(img_dist, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

        # Reconstruct distorted image
        with torch.no_grad():
            reconstructed = model(tensor_dist).cpu().numpy()[0, 0]  # shape (128,128)

        # Load all original fingerprints as array + keep paths
        originals_folder_full = os.path.join(self.resources_path, originals_folder)
        files = glob(os.path.join(originals_folder_full, '*.PNG')) + glob(os.path.join(originals_folder_full, '*.png'))
        if len(files) == 0:
            raise ValueError("No originals found for matching.")

        originals = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128)).astype(np.float32) / 255.0
            originals.append((f, img[..., np.newaxis]))  # keep path + image

        best_score = -1.0
        best_match_path = None

        for fpath, orig in originals:
            orig_resized = orig[..., 0]  # shape (128,128)
            score = ssim(reconstructed, orig_resized, data_range=1.0)
            if score > best_score:
                best_score = score
                best_match_path = fpath
        return best_match_path, best_score * 100

def load_images_from_folder(folder, img_size=(128, 128), return_dict=False):
    """
    If return_dict==True -> returns {filename_without_ext: image(H,W,1)}
    Otherwise returns numpy array of shape (N,H,W,1)
    """
    X = {} if return_dict else []
    handled_count = 0
    # accept both .BMP and .bmp
    files = glob(os.path.join(folder, '*.BMP')) + glob(os.path.join(folder, '*.bmp'))
    if len(files) == 0:
        print(f"Warning: no BMP files found in {folder}")

    for img_path in files:
        handled_count += 1
        print(f"handled images path: {folder} ---> count {handled_count}/{len(files)}", end="\r")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"\nWarning: failed to read {img_path}. Skipping.")
            continue
        try:
            img = cv2.resize(img, img_size)
        except Exception as e:
            print(f"\nWarning: failed to resize {img_path}: {e}. Skipping.")
            continue

        img = img.astype(np.float32) / 255.0
        img = img[..., np.newaxis]

        if return_dict:
            key = os.path.splitext(os.path.basename(img_path))[0]  # full filename without extension
            X[key] = img
        else:
            X.append(img)

    print("\n")
    return X if return_dict else np.array(X, dtype=np.float32)


class CnnModel(TrainingModel):
    
    def __init__(self, **args):
        super().__init__(**args)

class FingerprintCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # output 0-1
        return x
