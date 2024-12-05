import numpy as np
import json
import torch
import torchvision.transforms as transforms
from transformers import BertTokenizerFast, BertModel
import timm
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import random
import re
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

nltk.download('punkt')
nltk.download('punkt_tab')

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    image_encoder.train()
    text_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)

    image_feature_dim = 768
    image_projection = nn.Sequential(
        nn.Linear(1000, image_feature_dim),
        nn.Dropout(0.1)  # Dropout to prevent overfitting
    ).to(device)

    class TextDecoder(nn.Module):
        def __init__(self, vocab_size, feature_dim, hidden_dim, n_heads, n_layers, max_len=64):
            super(TextDecoder, self).__init__()
            self.embedding = nn.Embedding(vocab_size, feature_dim)
            self.position_embedding = nn.Parameter(torch.randn(1, max_len, feature_dim))
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_heads)
            self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers)
            self.fc = nn.Linear(feature_dim, vocab_size)
            self.max_len = max_len
            self.dropout = nn.Dropout(0.1)  # Dropout to prevent overfitting

        def forward(self, image_features, text_input, tgt_mask=None):
            text_embeddings = self.embedding(text_input) + self.position_embedding[:, :text_input.size(1), :]
            text_embeddings = self.dropout(text_embeddings)  # Apply dropout
            image_features = image_features.unsqueeze(1).repeat(1, text_input.size(1), 1)  # (batch_size, seq_len, feature_dim)
            text_output = self.transformer_decoder(text_embeddings.permute(1, 0, 2), image_features.permute(1, 0, 2), tgt_mask=tgt_mask)
            output = self.fc(text_output.permute(1, 0, 2))
            return output

    vocab_size = 30000
    hidden_dim = 768
    n_heads = 8
    n_layers = 4

    text_decoder = TextDecoder(vocab_size, image_feature_dim, hidden_dim, n_heads, n_layers).to(device)

    coco_ab_json_path = './coco_ab_v1_0.json'
    if not os.path.exists(coco_ab_json_path):
        raise FileNotFoundError(f"The file {coco_ab_json_path} does not exist. Please check the path.")

    with open(coco_ab_json_path, 'r') as f:
        coco_ab_data = json.load(f)

    train_data, temp_data = train_test_split(coco_ab_data[:5000], test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    class COCOABDataset(torch.utils.data.Dataset):
        def __init__(self, coco_data, transform=None):
            self.data = [item for item in coco_data if 'actionHistories' in item or 'categoryHistories' in item]
            self.transform = transform
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image_url = item.get('url', None)

            img = Image.new('RGB', (224, 224), color='black')  
            if image_url:
                if image_url.startswith('/') or image_url.startswith('./'):
                    try:
                        img = Image.open(image_url).convert('RGB')
                    except Exception as e:
                        print(f"Error loading local image: {image_url}, Error: {e}")
                elif re.match(r'(http|https)://', image_url):
                    try:
                        response = requests.get(image_url)
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                    except requests.exceptions.RequestException as e:
                        print(f"Error loading image from URL: {image_url}, Error: {e}")

            if self.transform:
                img = self.transform(img)

            click_info_text = "No specific action recorded."
            if 'actionHistories' in item and len(item['actionHistories']) > 0:
                action = item['actionHistories'][0]
                click_info_text = f"Action taken at point ({action['pointTo']['x']}, {action['pointTo']['y']}) of type {action['iconType']}."
            elif 'categoryHistories' in item and len(item['categoryHistories']) > 0:
                categories = [history['categoryName'] for history in item['categoryHistories']]
                click_info_text = "Categories involved: " + ", ".join(categories)

            tokenized_text = self.tokenizer(click_info_text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')

            return img.to(device), {key: val.to(device) for key, val in tokenized_text.items()}, click_info_text

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = COCOABDataset(train_data, transform=transform)
    val_dataset = COCOABDataset(val_data, transform=transform)

    # Increase batch size to ensure more negative pairs
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0, negative_weight=0.02):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin
            self.negative_weight = negative_weight

        def forward(self, image_features, text_features):
            image_features = nn.functional.normalize(image_features, p=2, dim=1) * 0.2
            text_features = nn.functional.normalize(text_features, p=2, dim=1) * 0.2

            similarity_matrix = torch.matmul(image_features, text_features.T)
            positive_pairs = torch.diag(similarity_matrix)
            negative_pairs = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(similarity_matrix.device) * 1e6

            positive_loss = 1 - positive_pairs
            negative_loss = torch.clamp(self.margin - negative_pairs, min=0).mean() * self.negative_weight

            loss = positive_loss.mean() + negative_loss
            return loss

    base_weight = 0.005
    growth_rate = 0.001
    contrastive_loss = ContrastiveLoss(margin=0.2, negative_weight=base_weight).to(device)
    caption_loss_fn = nn.CrossEntropyLoss()

    # Switch to AdamW optimizer and add learning rate scheduler
    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(image_projection.parameters()) + list(text_decoder.parameters()), lr=1e-5, weight_decay=0.01
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / 2000, 1.0))  # Extended warmup scheduler for more gradual learning rate increase
    num_epochs = 10
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))  # Cosine annealing scheduler

    train_losses = []
    val_losses = []
    bleu_scores = []

    alpha = 0.1
    beta = 5.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} Training:")
        epoch_loss = 0

        text_decoder.train()
        image_encoder.train()
        text_encoder.train()

        optimizer.zero_grad()

        # Adjust negative weight for the current epoch
        contrastive_loss.negative_weight = base_weight  # Fixed weight to maintain stability

        for batch_idx, batch in enumerate(train_loader):
            images, tokenized_texts, click_info_texts = batch
            images = images.to(device)
            tokenized_texts = {key: val.squeeze(1).to(device) for key, val in tokenized_texts.items()}

            image_features = image_encoder(images).view(images.size(0), -1)
            image_features = nn.functional.normalize(image_projection(image_features), p=2, dim=1)  # L2 normalization applied
            text_features = nn.functional.normalize(text_encoder(**tokenized_texts).last_hidden_state[:, 0, :], p=2, dim=1)  # L2 normalization applied

            target_texts = tokenized_texts['input_ids'][:, 1:]
            decoder_input = tokenized_texts['input_ids'][:, :-1]

            decoder_output = text_decoder(image_features, decoder_input)
            caption_loss_value = caption_loss_fn(decoder_output.reshape(-1, vocab_size), target_texts.reshape(-1))

            contrastive_loss_value = contrastive_loss(image_features, text_features)
            total_loss = alpha * contrastive_loss_value + beta * caption_loss_value

            

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += total_loss.item()

            # Log generated captions for monitoring
            if batch_idx % 50 == 0:
                generated_texts = decoder_output.argmax(dim=-1).cpu().numpy()
                candidate_texts = [bert_tokenizer.decode(generated_text, skip_special_tokens=True) for generated_text in generated_texts]
                print(f"Batch {batch_idx}, Generated Captions: {candidate_texts[:2]}")

            # Step the warmup scheduler if in warmup phase
            scheduler.step() if batch_idx < 2000 else cosine_scheduler.step()

            # Step the warmup scheduler if in warmup phase
            scheduler.step() if batch_idx < 2000 else cosine_scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Validation at the end of each epoch
        val_loss = 0
        val_bleu_score = 0
        text_decoder.eval()
        image_encoder.eval()
        text_encoder.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, tokenized_texts, click_info_texts = batch
                images = images.to(device)
                tokenized_texts = {key: val.squeeze(1).to(device) for key, val in tokenized_texts.items()}

                image_features = image_encoder(images).view(images.size(0), -1)
                image_features = nn.functional.normalize(image_projection(image_features), p=2, dim=1)  # L2 normalization applied
                text_features = nn.functional.normalize(text_encoder(**tokenized_texts).last_hidden_state[:, 0, :], p=2, dim=1)  # L2 normalization applied

                target_texts = tokenized_texts['input_ids'][:, 1:]
                decoder_input = tokenized_texts['input_ids'][:, :-1]

                decoder_output = text_decoder(image_features, decoder_input)
                caption_loss_value = caption_loss_fn(decoder_output.reshape(-1, vocab_size), target_texts.reshape(-1))

                val_loss += caption_loss_value.item()

                generated_texts = decoder_output.argmax(dim=-1).cpu().numpy()
                reference_texts = [nltk.word_tokenize(click_info) for click_info in click_info_texts]
                candidate_texts = [bert_tokenizer.decode(generated_text, skip_special_tokens=True) for generated_text in generated_texts]

                for ref, cand in zip(reference_texts, candidate_texts):
                    from nltk.translate.bleu_score import SmoothingFunction
                    smoothing_function = SmoothingFunction().method1
                    val_bleu_score += sentence_bleu([ref], nltk.word_tokenize(cand), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)  # BLEU-1
                    val_bleu_score += sentence_bleu([ref], nltk.word_tokenize(cand), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)  # BLEU-2
                    val_bleu_score += sentence_bleu([ref], nltk.word_tokenize(cand), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)  # BLEU-3
                    val_bleu_score += sentence_bleu([ref], nltk.word_tokenize(cand), smoothing_function=smoothing_function)  # BLEU-4

            avg_val_loss = val_loss / len(val_loader)
            avg_bleu_score = (val_bleu_score / (4 * len(val_loader.dataset)))  # Average of BLEU-1, BLEU-2, BLEU-3, and BLEU-4

            val_losses.append(avg_val_loss)
            bleu_scores.append(avg_bleu_score)

            print(f"Validation completed. Average Loss: {avg_val_loss:.4f}, Average BLEU Score: {avg_bleu_score:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, num_epochs + 1), bleu_scores, label='BLEU Score', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score over Epochs')
    plt.legend()
    plt.savefig('bleu_score.png')
    plt.show()
