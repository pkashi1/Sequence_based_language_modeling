import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class VanillaRNNModel(nn.Module):
    # the main language model with embedding and linear layers and functionality.
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(VanillaRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=3)  # pad_id = 3
        self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        #add the dropiout here
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
    def forward(self, input_ids, hidden=None):
        emb = self.dropout(self.embedding(input_ids))#embeddings
        output, hidden = self.rnn(emb, hidden)
        logits = self.fc(self.dropout(output))
        return logits, hidden
    def predict_next_token(self, logits, temperature=1.0):
        logits = logits[:, -1, :] / temperature#logits at final position
        # print(logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.squeeze(1)
    def generate_sequence(self, tokenizer, prompt, max_len=50, temperature=1.0, device='cuda'):
        self.eval()
        input_ids = tokenizer.encode(prompt)[:tokenizer.get_piece_size()]#ncode the input into tokenIDs
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated = input_ids.copy()
        with torch.no_grad():
            hidden = None
            for _ in range(max_len):
                logits, hidden = self(input_tensor, hidden)
                next_token = self.predict_next_token(logits, temperature=temperature).item()# use the temp sampling for prediction
                if next_token == tokenizer.eos_id():#stop the generation is EOS is reached.
                    break
                generated.append(next_token)#token addition to sequence
                input_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)#next input for next time step
        return tokenizer.decode(generated)#3 decode the tokens

class JSONLTextDataset(Dataset):#preprocessing of the jsonl files
    def __init__(self, path, tokenizer, seq_len=128):
        self.sequences = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)#get the each line from the json file.
                text = obj['prompt'] + obj['completion']
                tokens = tokenizer.encode(text)# tokenizes the text
                if len(tokens) <= 1:
                    continue
                    # break
                tokens = tokens + [tokenizer.eos_id()]
                for i in range(0, len(tokens) - seq_len,seq_len):
                    seq = tokens[i:i+seq_len+1]
                    self.sequences.append(seq)
    def __len__(self):#number of tokens availbles
        return len(self.sequences)
    def __getitem__(self, idx):
        #split the sequence into input and target.
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids
def collate_fn(batch):
    # Unzip inputs and targets from the batch
    inputs, targets = zip(*batch)
    #pdding the inputs and targets to normaize.
    inputs = pad_sequence(inputs, batch_first=True, padding_value=3)
    targets = pad_sequence(targets, batch_first=True, padding_value=3)
    return inputs, targets
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            logits = logits.view(-1, model.vocab_size)
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            mask = targets != 3
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)
def compute_bleu(model, dataset, tokenizer, device, num_samples=100):
    model.eval()
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for i in range(min(num_samples, len(dataset))):
        prompt, target = dataset[i]
        prompt_text = tokenizer.decode(prompt.tolist())
        target_text = tokenizer.decode(target.tolist())
        generated_text = model.generate_sequence(tokenizer, prompt_text, max_len=50, temperature=1.0, device=device)
        reference = [target_text.split()]
        hypothesis = generated_text.split()
        bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)
def train(model, train_loader, val_loader, epochs, lr, device, model_path='vanillarnn1.pt', patience=3):
    model.to(device)#move the model to the device.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=3)#cross entropy loss ignoring the padding index 3.
    
    best_val_loss = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            logits = logits.view(-1, model.vocab_size)
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, PPL = {val_ppl:.2f}")
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, no_improve = val_loss, 0
            torch.save(model.state_dict(), model_path)
        elif (no_improve := no_improve + 1) >= patience:
            print("Early stopping triggered.")
            break
    return train_losses, val_losses

def main():
    model_path = "RNN/vanillarnn1.pt"
    seq_len = 128
    batch_size = 128
    lr = 0.001
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("bpe_tokenizer.model")
    vocab_size = tokenizer.get_piece_size()
    train_data = JSONLTextDataset("data/train.jsonl", 
                                  tokenizer, 
                                  seq_len=seq_len)
    val_data = JSONLTextDataset("data/test.jsonl", 
                                tokenizer, 
                                seq_len=seq_len)
    train_loader = DataLoader(train_data, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_data, 
                            batch_size=batch_size, 
                            shuffle=False,
                              collate_fn=collate_fn)
    model = VanillaRNNModel(vocab_size)
    train_losses, val_losses = train(model, 
                                     train_loader, 
                                     val_loader, 
                                     epochs,
                                       lr, 
                                       device, 
                                       model_path)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("RNN/vanillarnn1.png")
    plt.show()
    model.load_state_dict(torch.load(model_path))
    prompts = ["Which do you prefer? Dogs or cats?", 
               "Hey I am a football player, I love playing football"]
    for prompt in prompts:
        output = model.generate_sequence(tokenizer, prompt, max_len=50, temperature=0.8, device=device)
        print(f"\nPrompt: {prompt}\nGenerated: {output}")
    _, val_ppl = evaluate(model, val_loader, nn.CrossEntropyLoss(ignore_index=3), device)
    bleu_score = compute_bleu(model, val_data, tokenizer, device)
    print(f"Validation Perplexity: {val_ppl:.2f}")
    print(f"Average BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    main()
