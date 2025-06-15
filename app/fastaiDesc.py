# %%capture
# Install all dependencies
!pip install torch torchvision transformers pandas numpy scikit-learn tqdm opencv-python-headless pillow fastai -q
!pip install git+https://github.com/openai/CLIP.git -q
# Download and unzip data
!gdown 'https://docs.google.com/uc?export=download&id=1uftdZXcf00X2MQVb8UUcRMaIor7Pfd1w' -O myfile.zip -q
!unzip -q myfile.zip
!rm myfile.zip

# Cell 1: Imports and Global Setup
import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm.auto import tqdm

# Fastai imports
from fastai.vision.all import *
from fastai.text.all import *

import clip

# Set device (fastai handles this automatically)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPU(s)!")

# Cell 2: BLIP Captioner (Utility Class)
class BLIPCaptioner:
    def __init__(self, device):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def generate_description(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=50)
            return self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return ""

# Cell 3: Model Architecture
class CrossAttentionMemeClassifier(nn.Module):
    def __init__(self, num_classes, text_model_name='facebook/xglm-564M', ablation_config=None):
        super().__init__()
        self.config = ablation_config if ablation_config else {}

        # Image Encoder (CLIP)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip = self.clip_model.visual.float()
        self.image_proj = nn.Linear(512, 768)

        # Text Encoder
        text_model_name = self.config.get('text_encoder', 'facebook/xglm-564M')
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 768)

        # Cross-Attention
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, texts):
        input_ids, attention_mask = texts

        # Image features
        with torch.no_grad():
            image_features = self.clip(images.to(torch.float32))

        # Ablation: Text-only or CLIP-only
        if self.config.get('text_only', False) or self.config.get('clip_only', False):
            image_features = torch.zeros_like(image_features)

        image_features = self.image_proj(image_features)

        # Text features
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_proj(text_output.last_hidden_state)

        # Ablation: Image-only
        if self.config.get('image_only', False):
            text_features = torch.zeros_like(text_features)

        # Cross-attention
        if self.config.get('cross_attention', True):
            attn_image, _ = self.image_attention(query=image_features.unsqueeze(1), key=text_features, value=text_features)
            attn_text, _ = self.text_attention(query=text_features, key=image_features.unsqueeze(1), value=image_features.unsqueeze(1))
            image_rep = attn_image.squeeze(1)
            text_rep = torch.mean(attn_text, dim=1)
        else: # No cross-attention, just use original features
            image_rep = image_features
            text_rep = torch.mean(text_features, dim=1)

        combined = torch.cat([image_rep, text_rep], dim=1)
        return self.classifier(combined)


# Cell 4: Data Pre-processing and Experiment Runner
def get_combined_text(row, use_blip):
    caption = row['Captions']
    if not use_blip: return caption
    description = row['Description']
    return f"{caption} [SEP] {description}"

# In Cell 4

def run_experiment(train_df, val_df, test_df, task, ablation_config, class_weights):
    print(f"\n--- Running Experiment with Config: {ablation_config} ---")

    # 1. Prepare Data based on ablation config
    use_blip = ablation_config.get('blip_descriptions', True)
    train_df['combined_text'] = train_df.apply(lambda row: get_combined_text(row, use_blip), axis=1)
    val_df['combined_text'] = val_df.apply(lambda row: get_combined_text(row, use_blip), axis=1)
    test_df['combined_text'] = test_df.apply(lambda row: get_combined_text(row, use_blip), axis=1)

    combined_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    # 2. Setup fastai DataBlock
    class HFTokenizer(Transform):
        def __init__(self, tokenizer): self.tokenizer = tokenizer
        def encodes(self, x:str):
            toks = self.tokenizer(x, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            return (toks['input_ids'].squeeze(), toks['attention_mask'].squeeze())

    text_model_name = ablation_config.get('text_encoder', 'facebook/xglm-564M')
    hf_tok = AutoTokenizer.from_pretrained(text_model_name)
    text_block = TransformBlock(type_tfms=HFTokenizer(hf_tok))

    multimodal_dblock = DataBlock(
        blocks=(ImageBlock, text_block, CategoryBlock),
        get_x=[ColReader('img_path'), ColReader('combined_text')],
        get_y=ColReader('Labels'),
        splitter=ColSplitter('is_valid'),
        item_tfms=Resize(224)
    )
    # SOLUTION 1: Reduce the batch size here
    dls = multimodal_dblock.dataloaders(combined_df, bs=16)

    # 3. Create Model and Learner
    num_classes = 2 if task == 'task1' else 4
    model = CrossAttentionMemeClassifier(num_classes, ablation_config=ablation_config)

    loss_func = CrossEntropyLossFlat(weight=class_weights.to(device))

    f1_macro = F1Score(average='macro')
    recall_macro = Recall(average='macro')
    metrics = [accuracy, f1_macro, recall_macro]

    # SOLUTION 2: Add GradientAccumulation callback
    # This will accumulate gradients over 2 mini-batches, simulating a batch size of 16*2=32
    callbacks = [
        GradientAccumulation(n_acc=2),
        SaveModelCallback(monitor='f1_score', comp=np.greater),
        EarlyStoppingCallback(monitor='f1_score', comp=np.greater, patience=2)
    ]

    learn = Learner(dls, model, loss_func=loss_func, metrics=metrics, cbs=callbacks).to_fp16()

    # 4. Train the Model
    learn.fine_tune(5, base_lr=2e-5)

    # 5. Evaluate and return F1 score
    test_dl = dls.test_dl(test_df, with_labels=True)
    preds, targs, _ = learn.get_preds(dl=test_dl)

    report = classification_report(targs.numpy(), preds.argmax(dim=1).numpy(), output_dict=True)
    return report['macro avg']['f1-score']


# Cell 5: Main Execution Block
if __name__ == "__main__":
    TASK = 'task2'
    DATA_DIR = "./BHM/Files"
    MEMES_DIR = "./BHM/Memes"

    # 1. Load Data
    train_df = pd.read_excel(os.path.join(DATA_DIR, f"train_{TASK}.xlsx"))
    val_df = pd.read_excel(os.path.join(DATA_DIR, f"valid_{TASK}.xlsx"))
    test_df = pd.read_excel(os.path.join(DATA_DIR, f"test_{TASK}.xlsx"))

    # Add image paths
    for df in [train_df, val_df, test_df]:
        df['img_path'] = df.apply(lambda row: os.path.join(MEMES_DIR, row['image_name']), axis=1)

    # 2. Pre-compute BLIP descriptions and cache them
    cache_file = 'descriptions_cache.pkl'
    if os.path.exists(cache_file):
        print("Loading cached BLIP descriptions...")
        descriptions = pd.read_pickle(cache_file)
    else:
        print("Generating BLIP descriptions (this will take a while)...")
        blip = BLIPCaptioner(device)
        # Combine all dfs to generate descriptions once
        all_df = pd.concat([train_df, val_df, test_df]).drop_duplicates(subset=['image_name'])
        desc_map = {row['image_name']: blip.generate_description(row['img_path']) for _, row in tqdm(all_df.iterrows(), total=len(all_df))}
        descriptions = train_df['image_name'].map(desc_map)
        pd.to_pickle(descriptions, cache_file)
    train_df['Description'] = descriptions

    # Map descriptions to validation and test sets
    desc_map = train_df.set_index('image_name')['Description'].to_dict()
    val_df['Description'] = val_df['image_name'].map(desc_map)
    test_df['Description'] = test_df['image_name'].map(desc_map)

    # 3. Encode Labels and Calculate Class Weights for Imbalance
    label_map = {'TI': 0, 'TC': 1, 'TO': 2, 'TS': 3} if TASK == 'task2' else {'non-hate': 0, 'hate': 1}
    for df in [train_df, val_df, test_df]:
        df['Labels'] = df['Labels'].map(label_map)
        df['is_valid'] = df.index.isin(val_df.index)

    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_df['Labels']), y=train_df['Labels'].values), dtype=torch.float32)

    # 4. Define and Run Ablation Studies
    baseline_f1 = run_experiment(train_df.copy(), val_df.copy(), test_df.copy(), TASK, {}, class_weights)
    print(f"\nBaseline F1 Score: {baseline_f1:.4f}")

    # ablation_configs = [
    #     {'name': 'No BLIP Descriptions', 'config': {'blip_descriptions': False}},
    #     {'name': 'No Cross-Attention', 'config': {'cross_attention': False}},
    #     {'name': 'Text Only', 'config': {'text_only': True}},
    #     {'name': 'Image Only', 'config': {'image_only': True}},
    #     {'name': 'mBERT instead of XGLM', 'config': {'text_encoder': 'bert-base-multilingual-cased'}}
    # ]

    # results = [{'Component': 'Baseline (Full Model)', 'F1 Score': f"{baseline_f1:.4f}", 'Degradation': '0.0%'}]

    # for study in ablation_configs:
    #     f1 = run_experiment(train_df.copy(), val_df.copy(), test_df.copy(), TASK, study['config'], class_weights)
    #     degradation = (baseline_f1 - f1) / baseline_f1 * 100
    #     results.append({'Component': study['name'], 'F1 Score': f"{f1:.4f}", 'Degradation': f"{degradation:.1f}%"})

    # # 5. Print Final Results Table
    # print("\n--- Ablation Study Final Results ---")
    # results_df = pd.DataFrame(results)
    # print(results_df.to_string(index=False))
