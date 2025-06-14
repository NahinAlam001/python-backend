# %%capture
!pip install torch torchvision transformers pandas numpy scikit-learn tqdm opencv-python-headless pillow
!pip install fastai
!pip install git+https://github.com/openai/CLIP.git
!gdown 'https://docs.google.com/uc?export=download&id=1uftdZXcf00X2MQVb8UUcRMaIor7Pfd1w' -O myfile.zip
!unzip -q myfile.zip
!rm myfile.zip

# Cell 1: Imports and Device Setup
import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from sklearn.metrics import classification_report

# Fastai imports
from fastai.vision.all import *
from fastai.text.all import *

import clip

# Set device (fastai handles this automatically, but good for confirmation)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPU(s)!")

# Cell 2: Model Architecture (Slightly adapted for fastai)
class CrossAttentionMemeClassifier(nn.Module):
    def __init__(self, num_classes, text_model_name='facebook/xglm-564M'):
        super().__init__()
        # Image encoder (CLIP)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip = self.clip_model.visual.float()
        self.image_proj = nn.Linear(512, 768)

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 768)

        # Cross-attention layers
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # CORRECTED VERSION
    def forward(self, images, texts):
        # The inputs are now passed directly as arguments
        input_ids, attention_mask = texts

        # Image features
        with torch.no_grad():
            image_features = self.clip(images.to(torch.float32))
        image_features = self.image_proj(image_features)

        # Text features
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_proj(text_output.last_hidden_state)

        # Cross-attention
        attn_image, _ = self.image_attention(
            query=image_features.unsqueeze(1),
            key=text_features,
            value=text_features
        )
        attn_text, _ = self.text_attention(
            query=text_features,
            key=image_features.unsqueeze(1),
            value=image_features.unsqueeze(1)
        )

        # Pooling and combining
        image_rep = attn_image.squeeze(1)
        text_rep = torch.mean(attn_text, dim=1)
        combined = torch.cat([image_rep, text_rep], dim=1)

        return self.classifier(combined)

# Cell 3: Utilities
def get_img_path(row, memes_dir):
    return os.path.join(memes_dir, row['image_name'])

def compute_metrics_fastai(task):
    def _compute_metrics(y_pred, y_true):
        # Convert predictions to class labels
        if task == 'task1':
            preds = (torch.sigmoid(y_pred) > 0.5).cpu().numpy().astype(int).squeeze()
        else:
            preds = torch.argmax(y_pred, dim=1).cpu().numpy()

        labels = y_true.cpu().numpy()
        report = classification_report(labels, preds, output_dict=True)
        return report['accuracy']
    return _compute_metrics

# Cell 4: Main Execution
if __name__ == "__main__":
    # Configuration
    TASK = 'task2'  # Change to 'task1' for binary classification
    DATA_DIR = "./BHM/Files"
    MEMES_DIR = "./BHM/Memes"
    BATCH_SIZE = 16
    TEXT_MODEL_NAME = 'facebook/xglm-564M'
    NUM_EPOCHS = 10
    LR = 2e-5

    # 1. Load and Prepare DataFrames
    train_df = pd.read_excel(os.path.join(DATA_DIR, f"train_{TASK}.xlsx"))
    val_df = pd.read_excel(os.path.join(DATA_DIR, f"valid_{TASK}.xlsx"))
    test_df = pd.read_excel(os.path.join(DATA_DIR, f"test_{TASK}.xlsx"))

    # Add a column to mark validation set for the splitter
    train_df['is_valid'] = False
    val_df['is_valid'] = True
    combined_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    # Add full image path to dataframes
    for df in [combined_df, test_df]:
        df['img_path'] = df.apply(lambda row: get_img_path(row, MEMES_DIR), axis=1)

    # 2. Setup fastai DataBlock
    # Create a custom tokenizer that outputs the required tuple of tensors
    class HFTokenizer(Transform):
        def __init__(self, tokenizer): self.tokenizer = tokenizer
        def encodes(self, x:str):
            toks = self.tokenizer(x, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            return (toks['input_ids'].squeeze(), toks['attention_mask'].squeeze())

    hf_tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # FIX: Wrap the custom transform in a TransformBlock
    text_block = TransformBlock(type_tfms=HFTokenizer(hf_tok))

    multimodal_dblock = DataBlock(
        # FIX: Pass ImageBlock type and the created text_block
        blocks=(ImageBlock, text_block, CategoryBlock),
        get_x=[ColReader('img_path'), ColReader('Captions')],
        get_y=ColReader('Labels'),
        splitter=ColSplitter('is_valid'),
        item_tfms=Resize(224)
    )

    dls = multimodal_dblock.dataloaders(combined_df, bs=BATCH_SIZE)

    # 3. Create the Model and Learner
    num_classes = 2 if TASK == 'task1' else 4
    model = CrossAttentionMemeClassifier(num_classes)

    # Define loss function based on the task
    loss_func = BCEWithLogitsLossFlat() if TASK == 'task1' else CrossEntropyLossFlat()

    # Create the Learner
    learn = Learner(dls, model, loss_func=loss_func, metrics=accuracy)

    # 4. Train the Model
    print("--- Starting Model Training ---")
    learn.fine_tune(NUM_EPOCHS, base_lr=LR)

    print("\n--- Training Complete ---")
    learn.save(f'best_model_{TASK}')

    # 5. Final Evaluation on Test Set
    print("\n--- Evaluating on Test Set ---")
    test_dl = dls.test_dl(test_df, with_labels=True)
    preds, targs, _ = learn.get_preds(dl=test_dl, with_decoded=True)

    if TASK == 'task1':
        pred_labels = (torch.sigmoid(preds) > 0.5).numpy().astype(int).squeeze()
        target_names = ['non-hate', 'hate']
    else:
        pred_labels = torch.argmax(preds, dim=1).numpy()
        target_names = ['TI', 'TC', 'TO', 'TS']

    print("\nFinal Test Results:")
    print(classification_report(targs.numpy(), pred_labels, target_names=target_names, digits=4))
