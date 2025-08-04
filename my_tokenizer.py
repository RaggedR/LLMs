import tiktoken
import torch
import unittest
import os
from torch.utils.data import Dataset, DataLoader

"""

GPT-2 was trained on the dataset called WebText which is approximately 40GB in size.
This dataset was Tokenized using Byte Pair Encoding (BPE)  
with a vocabulary of 50,257 tokens.
This is a proprietry dataset owned by OpenAI.

BPE begins with individual characters as the tokens, it then scans through the training
set and finds the most common pairs of characters. And turns these into new tokens.
It repeats this until it gets whole words or even phrases as a single token.

It is not necessarily true that if you encode and then decode you get exactly the same
string. Similarly if you decode and then encode you don't get exactly the same tokens.

"""





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
The class below inherits from the class unittest.TestCase.
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
The training data is a sequence of words from some text, and the label is the next word in the 
sequence. The context_size is the length of the original sequences. Here we have "stride" = 1.
"""
# context_generator.py (updated)
class SlidingContextWindowGenerator:
    """
    Generates fixed-length context-target pairs using a sliding window approach.
    Designed for language model training where:
      - Input: sequence of tokens [t_i, t_i+1, ..., t_i+n-1]
      - Target: next token sequence [t_i+1, t_i+2, ..., t_i+n]
    """
    def __init__(self, encoded_text, context_size, stride):
        """
        Args:
            encoded_text: List of token IDs
            context_size: Fixed length of context window (max_length)
            stride: Step size between consecutive windows
        """
        self.encoded_text = encoded_text
        self.context_size = context_size
        self.stride = stride

    def generate_samples(self):
        """
        Generates all (input, target) pairs using sliding window
        Returns:
            List of tuples: [(input_chunk, target_chunk), ...]
        """
        samples = []
        for i in range(0, len(self.encoded_text) - self.context_size, self.stride):
            input_chunk = self.encoded_text[i:i + self.context_size]
            target_chunk = self.encoded_text[i+1: i + self.context_size + 1]
            samples.append((input_chunk, target_chunk))
        return samples

# dataset.py
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)
        
        # Use generator to create training samples
        generator = SlidingContextWindowGenerator(
            encoded_text=token_ids,
            context_size=max_length,
            stride=stride
        )
        samples = generator.generate_samples()
        
        # Convert to tensors
        self.input_ids = [torch.tensor(x[0]) for x in samples]
        self.target_ids = [torch.tensor(x[1]) for x in samples]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# test_dataset.py (unchanged)
from torch.utils.data import DataLoader
# Assuming TikTokenizer is available
# from tokenizer import TikTokenizer

class TestFdGPTDatasetV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.txt = "Hello, world! This is a sample text for training GPT."
        cls.tokenizer = TikTokenizer()
        cls.max_length = 5
        cls.stride = 2
        cls.expected_tokens = cls.tokenizer.encode(cls.txt)
        
    def test_sample_content(self):
        dataset = GPTDatasetV1(self.txt, self.tokenizer, self.max_length, self.stride)
        input1, target1 = dataset[0]
        self.assertEqual(len(input1), self.max_length)
        self.assertEqual(len(target1), self.max_length)
        self.assertTrue(torch.equal(target1[:-1], input1[1:]))
        
        input2, target2 = dataset[1]
        self.assertEqual(len(input2), self.max_length)
        self.assertEqual(len(target2), self.max_length)
        self.assertEqual(input2[0], self.expected_tokens[self.stride])

    def test_dataloader_batching(self):
        dataset = GPTDatasetV1(self.txt, self.tokenizer, self.max_length, self.stride)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batches = list(loader)
        self.assertGreater(len(batches), 0)
        
        inputs, targets = batches[0]
        self.assertEqual(inputs.shape, (2, self.max_length))
        self.assertEqual(targets.shape, (2, self.max_length))
        self.assertTrue(torch.equal(targets[:, :-1], inputs[:, 1:]))

    def test_stride_effect(self):
        for stride in [1, 2, 3]:
            with self.subTest(stride=stride):
                dataset = GPTDatasetV1(self.txt, self.tokenizer, self.max_length, stride)
                input1, target1 = dataset[0]
                if stride > 1 and len(dataset) > 1:
                    input2, target2 = dataset[1]
                    self.assertEqual(input2[0], input1[stride])





"""
This code defines a custom PyTorch Dataset class called GPTDatasetV1 for preparing text 
data to train a GPT-style language model

Dataset is used to define custom datasets.
DataLoader can later be used to create iterable batches from the dataset.

This class inherits from torch.utils.data.Dataset.
It is designed to prepare input-target pairs for training a language model.

It uses a sliding window approach to split the tokenized text into overlapping sequences.
 
max_length: The maximum length of input sequences.
stride: How many tokens to move the sliding window forward.

The training sequences are stored as PyTorch tensors in self.input_ids and self.target_ids.   
     
"""


"""
The stride is how far you "skip along" to start the next training pair.
The max_length is the length of each input to the training pair.
EG: stride = 4 and max_length = 10 looks like
[a b c d e f g h i j] ----> [b c d e f g h i j k]
[f g h i j k l m n o] ----> [g h i j k l m n o p]
etc...
"""

class TestGPTDatasetV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup shared test fixtures"""
        cls.txt = "Hello, world! This is a sample text for training GPT."
        cls.tokenizer = TikTokenizer()
        cls.max_length = 5
        cls.stride = 2
        
        # Expected tokenized output (verify with your tokenizer)
        cls.expected_tokens = cls.tokenizer.encode(cls.txt)
        
    def test_sample_content(self):
        """Test the content of individual samples"""
        dataset = GPTDatasetV1(self.txt, self.tokenizer, self.max_length, self.stride)
        
        # Test first sample
        input1, target1 = dataset[0]
        self.assertEqual(len(input1), self.max_length)
        self.assertEqual(len(target1), self.max_length)
        self.assertTrue(torch.equal(target1[:-1], input1[1:]))  # Targets should be inputs shifted by 1
        #target1[:-1] refers to all elements of the target1 except the last one. 
        #We testing the fundamental autoregressive property of the dataset (predicting the next token), not the stride-based sampling.
        
        # Test second sample
        input2, target2 = dataset[1]
        self.assertEqual(len(input2), self.max_length)
        self.assertEqual(len(target2), self.max_length)
        #Verifies the sliding window is working correctly
        self.assertEqual(input2[0], self.expected_tokens[self.stride])

    def test_dataloader_batching(self):
        """Test the dataset works with DataLoader"""
        dataset = GPTDatasetV1(self.txt, self.tokenizer, self.max_length, self.stride)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batches = list(loader)
        self.assertGreater(len(batches), 0)
        
        # Verify first batch
        inputs, targets = batches[0]
        self.assertEqual(inputs.shape, (2, self.max_length))
        self.assertEqual(targets.shape, (2, self.max_length))
        
        # Verify shift between input and target
        self.assertTrue(torch.equal(targets[:, :-1], inputs[:, 1:]))

    def test_stride_effect(self):
        """Test different stride values produce correct samples"""
        for stride in [1, 2, 3]:
            with self.subTest(stride=stride):
                dataset = GPTDatasetV1(self.txt, self.tokenizer, self.max_length, stride)
                input1, target1 = dataset[0]
                if stride > 1:
                    input2, target2 = dataset[1]
                    self.assertEqual(input2[0], input1[stride])

if __name__ == "__main__":
    unittest.main()
    






"""

'''
Example useage
'''


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





DataLoader Setup:

    Wraps the dataset into an iterable loader.

    Batches sequences (e.g., batch_size=4 sequences per batch).

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

'''
Example useage
'''

text = "The quick brown fox jumps over the lazy dog."
dataloader = create_dataloader_v1(text, max_length=4, stride=1, batch_size=2)

for batch in dataloader:
    print(batch)  # Batched token IDs (shape: [batch_size, max_length])










