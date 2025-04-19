# Sequence_based_language_modeling
Sequential Deep Learning for Language Modeling with RNNs, LSTMs, and Transformers

##Environment Setup


Make sure you activate your environment before running any code:

```bash
conda env create -f my_env.yml
conda activate myenv_torch_cuda
```

## Run Instructions
The codes already have the bpe_tokenizer model and pretrained models loaded. If the tokenizer is not loaded:

```bash
python spe.py
```
### Vanilla RNN

```bash
python RNN/vanillarnn.py
```

- Output: Trained model `vanillarnn1.pt` and loss plot `vanillarnn1.png`

---

### LSTM
```bash
python LSTM/lstm2.py
```
- Output: Trained model `lstm_model_latest4.pt` and loss plot `lstm4.png`

### Transformer (Encoder-only)

```bash
python Transformer_encoder/encoder.py
```

- Output: Trained model `transformer_model_encoder_parimal1.pt` and loss plot `transformer_model_encoder_parimal1.png`

### To Run the models on HPC
You would only have to change the file names and resource alloaction IDs

```bash
sbatch submit.sh
```

## Notes

- All models use the same `bpe_tokenizer.model` trained using SentencePiece.
- Training includes:  
  - `CrossEntropyLoss`  
  - `AdamW` optimizer  
  - Batch size = 128  
  - Epochs = 30 with Early Stopping  
  - Learning Rate Scheduler  
- Generation uses temperature sampling for text diversity.

