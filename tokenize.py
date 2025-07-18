import tiktoken

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
