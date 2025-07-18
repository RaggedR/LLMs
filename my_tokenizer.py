import tiktoken
import torch

class TikTokenizer:
    def __init__(self, encoding_name="gpt2"):
        """Initialize tokenizer with specified encoding"""
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def encode(self, text):
        """Convert text to token IDs"""
        return self.encoding.encode(text)
    
    def decode(self, tokens):
        """Convert token IDs back to text"""
        return self.encoding.decode(tokens)

"""

GPT-2 was trained on the dataset called WebText which is approximately 40GB in size.
This dataset was Tokenized using Byte Pair Encoding (BPE)  
with a vocabulary of 50,257 tokens.
This is a proprietry dataset owned by OpenAI.

BPE begins with individual characters as the tokens, it then scans through the training
set and findz the most common pairs of characters. And turns these into new tokens.
It repeats this until it gets whole words or even phrases as a single token.

It is not necessarily true that if you encode and then decode you get exactly the same
string. Similarly if you decode and then encode you don't get exactly the same tokens.

"""

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

my_tokenizer = TikTokenizer()

enc_text = my_tokenizer.encode(raw_text)
print(len(enc_text))

context_size = 4

x = enc_text[:context_size]
y = enc_text[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_text[:i]
    desired = enc_text[i]

    print(context, "---->", desired)
    
for i in range(1, context_size+1):
    context = enc_text[:i]
    desired = enc_text[i]

    print(my_tokenizer.decode(context), "---->", my_tokenizer.decode([desired]))

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


"""
The stride is how far you "skip along" to start the next training pair.
The max_length is the length of each input to the training pair.
EG: stride = 4 and max_length = 10 looks like
[a b c d e f g h i j] ----> [b c d e f g h i j k]
[f g h i j k l m n o] ----> [g h i j k l m n o p]
etc...
"""

def create_dataloader_v1(txt, max_length=256, 
                         stride=128, batch_size=4, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    my_tokenizer = TikTokenizer()

    # Create dataset (function from torch)
    dataset = GPTDatasetV1(txt, my_tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader












