import os
import sentencepiece as spm
from typing import List, Tuple


def merge_text_files(data_dir: str, output_file: str):
    """
    Merge all .txt files from data_dir into output_file
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line.strip() + "\n")


def add_special_tokens(pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Insert <bos> and <eos> into prompts and completions
    """
    new_prompts, new_completions = [], []
    
    for prompt, completion in pairs:
        if prompt and prompt[0].isupper():
            prompt = "<bos> " + prompt
        if completion and (completion.endswith('.') or completion.endswith('?') or completion.endswith('!')):
            completion = completion + " <eos>"
        new_prompts.append(prompt)
        new_completions.append(completion)
        
    return new_prompts, new_completions


def train_tokenizer(corpus_file: str, prefix: str, vocab_size: int = 10000):
    """
    Trains a SentencePiece tokenizer with special tokens
    """
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=prefix,
        vocab_size=vocab_size,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"])
    )
    print("Tokenizer training complete!")
    print(f"{prefix}.model")
    print(f"{prefix}.vocab")


if __name__ == "__main__":
    DATA_DIR = "./data/raw"
    TOKENIZER_PREFIX = "bpe_tokenizer"
    VOCAB_SIZE = 10000
    CORPUS_FILE = "corpus.txt"
    merge_text_files(DATA_DIR, CORPUS_FILE)

    train_tokenizer(CORPUS_FILE, TOKENIZER_PREFIX, VOCAB_SIZE)