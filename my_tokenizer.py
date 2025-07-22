import tiktoken
import torch


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

import unittest
import tiktoken
import os
from torch.utils.data import Dataset, DataLoader

class TikTokenizer:
    def __init__(self, encoding_name="gpt2", allowed_special={"<|endoftext|>"}):
        """Initialize tokenizer with specified encoding"""
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def encode(self, text):
        """Convert text to token IDs"""
        return self.encoding.encode(text)
    
    def decode(self, tokens):
        """Convert token IDs back to text"""
        return self.encoding.decode(tokens)


""" 
This class inherits from the class unittest.TestCase.
assertEqual is method inherited from unittest.TestCase
All methods starting with test_ are run when we call unittest.main()
Output: OK means all tests were passed.
"""

class TestTikTokenizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.tokenizer = TikTokenizer()
    
    def test_encode(self):
        """Test encoding text to tokens"""
        text = "Hello world!"
        expected_tokens = [15496, 995, 0]  
        actual_tokens = self.tokenizer.encode(text)
        self.assertEqual(actual_tokens, expected_tokens, "Token encoding mismatch")
    
    def test_decode(self):
        """Test decoding tokens back to text"""
        tokens = [15496, 995, 0]  
        expected_text = "Hello world!"
        actual_text = self.tokenizer.decode(tokens)
        self.assertEqual(actual_text, expected_text, "Text decoding mismatch")


"""
Just some visualizations of what we want to do
"""


class TestTikTokenizerShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests"""
        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            cls.raw_text = f.read()
        cls.tokenizer = TikTokenizer()
        cls.enc_text = cls.tokenizer.encode(cls.raw_text)
        
        # Expected values from your output
        cls.expected_x = [40, 367, 2885, 1464]
        cls.expected_y = [367, 2885, 1464, 1807]
        cls.expected_pairs = [
            ([40], 367),
            ([40, 367], 2885),
            ([40, 367, 2885], 1464),
            ([40, 367, 2885, 1464], 1807)
        ]
        cls.expected_decoded_pairs = [
            ("I", " H"),
            ("I H", "AD"),
            ("I HAD", " always"),
            ("I HAD always", " thought")
        ]
    def test_context_windows(self):
        """Test the x and y context windows"""
        context_size = 4
        x = self.enc_text[:context_size]
        y = self.enc_text[1:context_size+1]
        
        self.assertEqual(x, self.expected_x, "Initial context window (x) mismatch")
        self.assertEqual(y, self.expected_y, "Shifted context window (y) mismatch")

    def test_token_prediction_pairs(self):
        """Test the token prediction pairs"""
        for i, (expected_context, expected_target) in enumerate(self.expected_pairs, 1):
            #Note that self.enc_text[:1] is a list containing the first elements of
            # the list self.enc_text.
            context = self.enc_text[:i]
            target = self.enc_text[i]
            with self.subTest(i=i):
                self.assertEqual(context, expected_context, f"Context at position {i} mismatch")
                self.assertEqual(target, expected_target, f"Target at position {i} mismatch")

    def test_decoded_prediction_pairs(self):
        """Test the decoded prediction pairs"""
        for i, (expected_context, expected_target) in enumerate(self.expected_decoded_pairs, 1):
            #Note that self.enc_text[:1] is a list containing the first elements of
            # the list self.enc_text.
            context = self.enc_text[:i]
            target = self.enc_text[i]
            with self.subTest(i=i):
                decoded_context = self.tokenizer.decode(context)
                decoded_target = self.tokenizer.decode([target])
                self.assertEqual(decoded_context, expected_context, f"Decoded context at position {i} mismatch")
                self.assertEqual(decoded_target, expected_target, f"Decoded target at position {i} mismatch")

if __name__ == "__main__":
    unittest.main()


"""
This code defines a custom PyTorch Dataset class called GPTDatasetV1 for preparing text data to train a GPT-style language model

Dataset is used to define custom datasets.
DataLoader can later be used to create iterable batches from the dataset.

This class inherits from torch.utils.data.Dataset.
It is designed to prepare input-target pairs for training a language model.

It uses a sliding window approach to split the tokenized text into overlapping sequences.
 
max_length: The maximum length of input sequences.
stride: How many tokens to move the sliding window forward.

The training sequences are stored as PyTorch tensors in self.input_ids and self.target_ids.   
     
"""



class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)
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
Example useage
"""


txt = "Hello, world! This is a sample text for training GPT."
tokenizer = TikTokenizer()
max_length = 5
stride = 2

dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch in loader:
    inputs, targets = batch
    print("Input:", inputs)
    print("Target:", targets)


"""
The stride is how far you "skip along" to start the next training pair.
The max_length is the length of each input to the training pair.
EG: stride = 4 and max_length = 10 looks like
[a b c d e f g h i j] ----> [b c d e f g h i j k]
[f g h i j k l m n o] ----> [g h i j k l m n o p]
etc...
"""

"""
DataLoader Setup:

    Wraps the dataset into an iterable loader.

    Batches sequences (e.g., batch_size=4 → 4 sequences per batch).

    Enables shuffling and parallel loading (num_workers).
    
drop_last=True ensures uniform batch sizes (critical for GPU training).

num_workers specifies how many subprocesses to use for:

    Loading data

    Preprocessing (tokenization, augmentation, etc.)

    Batching
    
Each worker:

    Loads data independently

    Applies transforms

    Returns batches via shared memory
    
Start with 0 for debugging (fewer multiprocessing issues)

Output is a PyTorch DataLoader, compatible with standard training loops:
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

"""
Example useage
"""

text = "The quick brown fox jumps over the lazy dog."
dataloader = create_dataloader_v1(text, max_length=4, stride=1, batch_size=2)

for batch in dataloader:
    print(batch)  # Batched token IDs (shape: [batch_size, max_length])










