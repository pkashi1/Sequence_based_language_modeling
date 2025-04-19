import json
import math
import os
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.sequences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj['prompt'] + obj['completion']
                    tokens = tokenizer.encode(text) + [tokenizer.eos_id()]
                    # print(f"Tokens: {tokens}")
                    if len(tokens) > 1:
                        for i in range(0, len(tokens) - seq_length, seq_length):
                            # print(len(seq_length),seq_length)
                            self.sequences.append(tokens[i:i + seq_length + 1])
                except json.JSONDecodeError:
                    continue
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # print(f"Sequence: {seq}")
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()# we use this embedding vector as it has multiplication factor of positionID
        #taking the postion ID to know the position and add the positonal embeddings to build the attention matrix
        self.dropout = nn.Dropout(dropout)
        position_matrix = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        position_matrix[:, 0::2] = torch.sin(position * div_term)
        position_matrix[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_matrix', position_matrix.unsqueeze(0))
    def forward(self, x):
        x = x + self.position_matrix[:, :x.size(1), :]
        return self.dropout(x)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                    nhead=nhead,
                                                    dim_feedforward=hidden_dim,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return self.output_layer(x)
    # def generate_text(self, prompt, max_len=50, temperature=0.7,top_k = 50,device = 'cuda'):
#     self.eval()
#     input_ids = self.tokenizer.encode(prompt)
#     for _ in range(max_len):
#         next_token = self.predict_next_token(input_ids)
#         input_ids.append(next_token)
#         if next_token == self.tokenizer.eos_token_id:
#             break
#     return self.tokenizer.decode(input_ids)
    def generate(self, tokenizer, prompt, max_len=50, temperature=1.0, device='cuda'):
        self.eval()
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated = input_ids.copy()
        with torch.no_grad():
            for _ in range(max_len):
                # predict_next_token():
                # self.eval 
                # with torch.no_grad():
                #     logits =  self.forward(inputids, )
                #     logits = logits[0, -1, :]
                #     probs = F.softmax(logits, dim=-1)
                #     next_token = torch.multinomial(probs, num_samples=1)
                #     return next_token.item()
                logits = self(input_tensor)
                logits = logits[:, -1, :] / temperature# once the model pretrained postional matrix is fixed 
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == tokenizer.eos_id():
                    break
                generated.append(next_token)
                input_tensor = torch.tensor([generated], dtype=torch.long).to(device)# 
        return tokenizer.decode(generated)
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
            non_pad = targets.view(-1) != 0
            total_tokens += non_pad.sum().item()
            total_loss += loss.item() * non_pad.sum().item()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)

def compute_bleu(model, dataset, tokenizer, device, num_samples=100):
    model.eval()
    smoothie = SmoothingFunction().method4
    scores = []
    for i in range(min(num_samples, len(dataset))):
        input_tensor, target_tensor = dataset[i]
        prompt_text = tokenizer.decode(input_tensor.tolist())
        target_text = tokenizer.decode(target_tensor.tolist())
        generated = model.generate(tokenizer,
                                   prompt_text,
                                   device=device)
        reference = [target_text.split()]
        
        hypothesis = generated.split()
        scores.append(sentence_bleu(reference,
                                    hypothesis,
                                    smoothing_function=smoothie))
    return sum(scores) / len(scores)

def train(model, train_loader, val_loader, device, epochs=30, lr=0.0005):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, PPL={val_ppl:.2f}")
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "Transformer_decoder/best_transformer_model.pt")
        else:
            patience_counter =patience_counter+ 1# incrementing the counter to see until when the model is not leanring
            if patience_counter >= 4:#doing the early stopping
                break
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig("Transformer_decoder/transformer_model_encoder_parimal1.png")
    plt.close()
    return model

def collate_fn(batch):
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    return x, y
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("bpe_tokenizer.model")
    train_dataset = TextDataset("data/train.jsonl", tokenizer)
    val_dataset = TextDataset("data/test.jsonl", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    model = TransformerModel(vocab_size=tokenizer.get_piece_size())
    model = train(model, train_loader, val_loader, device)
    for prompt in ["Which do you prefer? Dogs or cats?", "I am a football player I love playing football"]:
        print(f"Prompt: {prompt}")
        print(model.generate(tokenizer, prompt, device=device))
    val_loss, val_ppl = evaluate(model, val_loader, nn.CrossEntropyLoss(ignore_index=0), device)
    bleu = compute_bleu(model, val_dataset, tokenizer, device)
    print(f"Validation Perplexity: {val_ppl:.2f}")
    print(f"BLEU Score: {bleu:.4f}")

if __name__ == '__main__':
    main()
# he used decoder only model
# positional encoding is on moodle
# generate squaresubsequent mask(self,sz,devices)::
# from sympy.abc import x
# import torch.nn.functional as F

