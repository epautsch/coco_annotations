#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter

from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
nltk.download('punkt')


# In[2]:


image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


# In[3]:


class CaptionPreprocessor:
    def __init__(self, captions, vocab_threshold=5, max_caption_length=20):
        self.max_caption_length = max_caption_length

        all_tokens = [token for caption in captions for token in nltk.tokenize.word_tokenize(caption.lower())]
        counter = Counter(all_tokens)
        self.vocab = {token: idx for idx, (token, count) in enumerate(counter.items()) if count >= vocab_threshold}

        self.vocab['<pad>'] = len(self.vocab)
        self.vocab['<start>'] = len(self.vocab)
        self.vocab['<end>'] = len(self.vocab)
        self.vocab['<unk>'] = len(self.vocab)

        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

        self.captions_tokenized = self.tokenize_captions(captions)

    def preprocess(self, caption):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption_indices = [self.vocab['<start>']] + [self.vocab.get(token, self.vocab['<unk>']) for token in tokens] + [self.vocab['<end>']]

        if len(caption_indices) < self.max_caption_length:
            caption_indices += [self.vocab['<pad>']] * (self.max_caption_length - len(caption_indices))

        return caption_indices[:self.max_caption_length]

    def tokenize_captions(self, captions):
        return [self.preprocess(caption) for caption in captions]


# In[4]:


class CustomCocoDataset(Dataset):
    def __init__(self, coco_dataset, caption_preprocessor):
        self.coco_dataset = coco_dataset
        self.caption_preprocessor = caption_preprocessor

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        img, caption_list = self.coco_dataset[idx]
        caption = caption_list[0]
        preprocessed_caption = torch.tensor(self.caption_preprocessor.preprocess(caption))
        return img, preprocessed_caption


# In[6]:


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_layers, num_heads, mlp_dim, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, (224 // patch_size) * (224 // patch_size) + 1, embed_dim))

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.positional_encoding[:, :-1]
        for layer in self.transformer_layers:
            x = layer(x)

        return x


# In[7]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, :, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, :, 1::2] = torch.cos(pos * div_term)


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, mlp_dim, max_len=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, captions, memory):
        captions = self.embedding(captions) + self.positional_encoding.encoding[:, :captions.shape[1]]

        for layer in self.transformer_layers:
            captions = layer(captions, memory)

        logits = self.output_layer(captions)
        return logits


# In[8]:


class ImageCaptioningModel(nn.Module):
    def __init__(self, image_encoder, caption_decoder, embedding_size):
        super(ImageCaptioningModel, self).__init__()
        self.image_encoder = image_encoder
        self.caption_decoder = caption_decoder
        self.embedding_size = embedding_size

    def forward(self, images, captions):
        image_features = self.image_encoder(images)
        num_patches = (224 // 16) * (224 // 16)
        image_features_flattened = image_features.permute(1, 0, 2).reshape(-1, num_patches, self.embedding_size)

        print("Start token index:", caption_preprocessor.vocab['<start>'])

        start_token_embeddings = self.caption_decoder.embedding(torch.tensor([caption_preprocessor.vocab['<start>']], device=device)).repeat(image_features.shape[0], 1, 1) # Get the <start> token embedding and repeat it for the batch size
        image_features_summed = image_features_flattened.sum(dim=1).unsqueeze(1)
        memory = torch.cat([start_token_embeddings, image_features_summed], dim=1) # Concatenate the start token embeddings with the flattened image features

        memory = memory.transpose(0, 1)

        captions = captions.transpose(0, 1)

        output = self.caption_decoder(captions, memory)

        output = output.transpose(0, 1)

        return output


# In[ ]:


dataset = CocoCaptions(root='./coco/images',
                       annFile='./coco/annotations/captions_train2014.json',
                       transform=image_transform)
captions = [entry['caption'] for entry in dataset.coco.anns.values()]
caption_preprocessor = CaptionPreprocessor(captions)
custom_dataset = CustomCocoDataset(dataset, caption_preprocessor)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True, num_workers=4)


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_encoder = VisionTransformer(in_channels=3,
                                  patch_size=16,
                                  embed_dim=768,
                                  num_layers=12,
                                  num_heads=12,
                                  mlp_dim=3072,
                                  num_classes=768).to(device)
max_caption_index = max([max(caption) for caption in caption_preprocessor.captions_tokenized])
caption_decoder = TransformerCaptionDecoder(vocab_size=max_caption_index + 1,
                                            d_model=768,
                                            num_layers=6,
                                            num_heads=8,
                                            mlp_dim=2048).to(device)
embedding_size = 768
model = ImageCaptioningModel(image_encoder, caption_decoder, embedding_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=caption_preprocessor.vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, (images, captions) in enumerate(data_loader):
        images = images.to(device)
        captions_input = captions[:, :-1].to(device)
        captions_target = captions[:, 1:].to(device)

        print("Captions shape:", captions_input.shape)
        print("Memory shape:", images.shape)

        print("Max index in captions_input:", captions_input.max().item())

        print("Embedding layer num_embeddings:", model.caption_decoder.embedding.num_embeddings)

        optimizer.zero_grad()
        output = model(images, captions_input)

        print("Output shape:", output.shape)
        print("Captions target shape:", captions_target.shape)
        print("Output view shape:", output.reshape(-1, 24535).shape)
        print("Captions target view shape:", captions_target.view(-1).shape)

        loss = criterion(output.reshape(-1, 24535), captions_target.view(-1))
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Iteration: {i}, Loss: {loss.item()}')


# In[ ]:




