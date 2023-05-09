#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import os
import time
import random
import pickle

from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.nn.init as init
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# In[ ]:


image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


# In[ ]:


class CaptionPreprocessor:
    def __init__(self, captions, tokenizer, max_caption_length=14):
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length
        self.total_caption_length = 0
        self.total_squared_caption_length = 0

        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = '[CLS]'
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = '[SEP]'

        self.captions_tokenized = self.tokenize_captions(captions)

    def preprocess(self, caption):
        tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(caption) + [self.tokenizer.eos_token]
        caption_indices = self.tokenizer.convert_tokens_to_ids(tokens)
        caption_length = len(caption_indices)

        if caption_length < self.max_caption_length:
            caption_indices += [self.tokenizer.pad_token_id] * (self.max_caption_length - caption_length)

        self.total_caption_length += caption_length
        self.total_squared_caption_length += caption_length**2

        return caption_indices[:self.max_caption_length]

    def tokenize_captions(self, captions):
        return [self.preprocess(caption) for caption in captions]

    def get_average_caption_length(self):
        return self.total_caption_length / len(self.captions_tokenized)

    def get_caption_length_standard_deviation(self):
        mean = self.get_average_caption_length()
        variance = (self.total_squared_caption_length / len(self.captions_tokenized)) - mean**2
        std_dev = math.sqrt(max(0, variance))
        return std_dev


# In[ ]:


class CustomCocoDataset(Dataset):
    def __init__(self, coco_dataset, caption_preprocessor, num_captions=5):
        self.coco_dataset = coco_dataset
        self.caption_preprocessor = caption_preprocessor
        self.num_captions = num_captions

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        img, caption_list = self.coco_dataset[idx]

        # hits when using test dataset which doesn't have captions
        if not caption_list:
            return img, None

        selected_caption = random.choice(caption_list[:self.num_captions])
        preprocessed_caption = torch.tensor(self.caption_preprocessor.preprocess(selected_caption))
        return img, preprocessed_caption


# In[ ]:


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_layers, num_heads, mlp_dim, num_classes, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, (224 // patch_size) * (224 // patch_size) + 1, embed_dim))

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.positional_encoding[:, :-1]
        for layer in self.transformer_layers:
            x = layer(x)

        return x


# In[ ]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        max_len = x.size(1)
        encoding = torch.zeros(1, max_len, self.d_model, device=x.device, requires_grad=False)

        pos = torch.arange(0, max_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-torch.log(torch.tensor(10000.0, device=x.device)) / self.d_model))
        encoding[:, :, 0::2] = torch.sin(pos * div_term)
        encoding[:, :, 1::2] = torch.cos(pos * div_term)

        x = x.add(encoding)
        return x


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, auto_model, d_model, num_layers, num_heads, mlp_dim, dropout=0.0):
        super().__init__()

        self.auto_model = auto_model
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, mlp_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, self.auto_model.config.vocab_size)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, captions, memory):
        captions = self.auto_model.get_input_embeddings()(captions)
        captions = self.positional_encoding(captions).detach()

        memory = memory.transpose(0, 1)

        for layer in self.transformer_layers:
            captions = layer(captions, memory[:, :captions.size(1), :])

        logits = self.output_layer(captions)
        return logits


# In[ ]:


class ImageCaptioningModel(nn.Module):
    def __init__(self, image_encoder, caption_decoder):
        super(ImageCaptioningModel, self).__init__()
        self.image_encoder = image_encoder
        self.caption_decoder = caption_decoder
        self.start_token_index = caption_decoder.auto_model.config.bos_token_id or 0
        self.embedding_size = caption_decoder.auto_model.config.hidden_size
        self.image_feature_linear = nn.Linear(768, self.embedding_size)

    def forward(self, images, captions, teacher_forcing=True):
        image_features = self.image_encoder(images)
        num_patches = (224 // 16) * (224 // 16)
        # image_features_flattened = image_features.permute(1, 0, 2).reshape(-1, num_patches, self.embedding_size)

        start_token_tensor = torch.tensor([self.start_token_index], dtype=torch.long, device=images.device)
        start_token_embeddings = self.caption_decoder.auto_model.embeddings(start_token_tensor).repeat(image_features.shape[0], 1, 1) # getting start token embedding and repeating it for batch size
        image_features_summed = image_features.sum(dim=1).unsqueeze(1)
        image_features_summed = self.image_feature_linear(image_features_summed)
        memory = torch.cat([start_token_embeddings, image_features_summed], dim=1) # Concat the start token embeddings with the flattened image features
        memory = memory.transpose(0, 1)

        if teacher_forcing:
            captions_input = captions[:, :-1].to(device)
            captions_output = self.caption_decoder(captions_input, memory)
        else:
            captions_output = torch.zeros_like(captions).to(device)
            logits_output = torch.zeros((captions.size(0), captions.size(1), self.caption_decoder.auto_model.config.vocab_size), device=device)
            captions_output[:, 0] = start_token_tensor
            for t in range(1, captions.size(1)):
                captions_input = captions_output[:, :t].to(device)
                logits = self.caption_decoder(captions_input, memory[:, :t].clone())
                logits_output[:, t] = logits[:, -1]
                captions_output[:, t] = logits[:, -1].argmax(-1)

            captions_output = logits_output

        return captions_output


    def update_dropout_rate(self, new_dropout_rate):
        def update_dropout(module):
            if isinstance(module, nn.Dropout):
                module.p = new_dropout_rate

        self.caption_decoder.apply(update_dropout)


# In[ ]:


def use_teacher_forcing(teacher_forcing_ratio):
    return random.random() < float(teacher_forcing_ratio)


def train_one_epoch(model,
                    dataloader,
                    criterion,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    num_epochs,
                    avg_every,
                    learning_rates,
                    stepCounter=None,
                    teacher_forcing_ratio=1.0):
    torch.autograd.set_detect_anomaly(True)

    model.train()
    train_loss = 0
    last_x_losses = []

    progress_bar = tqdm(dataloader, desc='Training')

    for i, (images, captions) in enumerate(progress_bar):
        images = images.to(device)
        captions_input = captions.to(device)

        use_tf = use_teacher_forcing(teacher_forcing_ratio)

        optimizer.zero_grad()

        output = model(images, captions_input, teacher_forcing=use_tf)

        if use_tf:
            captions_target = captions[:, 1:-1].to(device)
        else:
            captions_target = captions[:, 1:].to(device)

        loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), captions_target.view(-1))
        loss.backward()

        learning_rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
        # step.addAstep() # use with other schedulers

        train_loss += loss.item()
        last_x_losses.append(loss.item())

        if i % avg_every == 0 and i != 0:
            avg_loss = sum(last_x_losses) / len(last_x_losses)
            print(f'Epoch: {epoch+1}/{num_epochs}, Iteration: {i}, Loss (last {avg_every} iterations): {avg_loss:.4f}')
            # progress_bar.set_postfix(avg_loss_last_x_iterations=f'last {avg_every}: {avg_loss:.4f}')
            last_x_losses = []

    return train_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc='Validating'):
            image = images.to(device)
            captions_input = captions[:, :-1].to(device)
            captions_target = captions[:, 1:].to(device)

            output = model(images, captions_input, teacher_forcing=False)

            captions_target = captions[:, 1:-1].to(device)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), captions_target.view(-1))

            val_loss += loss.item()
    return val_loss / len(dataloader)



# In[ ]:


class stepCounter:
    def __init__(self):
        self.steps = 0

    def addAstep(self):
        self.steps += 1


# In[ ]:


class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000, scaling_factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.scaling_factor = scaling_factor

    def step(self):
        self.current_step += 1
        lr = self.learning_rate()
        # print line used for debugging optimizer state on reload
        # print(f'NoamScheduler step: {self.current_step}, Learning rate: {lr}')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def learning_rate(self):
        arg1 = self.current_step ** -0.5
        arg2 = min(self.current_step * self.warmup_steps ** -1.5, 1)
        return (self.d_model ** -0.5) * min(arg1, arg2) * self.scaling_factor


# In[ ]:


class TeacherForcingScheduler:
    def __init__(self, max_epochs, initial_teacher_forcing_ratio=1.0, warmup_steps=4000):
        self.curr_epoch = 0
        self.max_epochs = max_epochs
        self.initial_teacher_forcing_ratio = initial_teacher_forcing_ratio
        self.curr_teacher_forcing_ratio = initial_teacher_forcing_ratio
        self.last_val_loss = float('inf')
        self.epochs_since_best_val_loss_set = 0
        self.warmup_steps = warmup_steps
        self.tf_history = []


    # Linear decay
    def step(self, val_loss, curr_sched_step):
        self.tf_history.append(self.curr_teacher_forcing_ratio)

        if val_loss < self.last_val_loss:
            self.epochs_since_best_val_loss_set = 0
        elif curr_sched_step > self.warmup_steps:
            self.epochs_since_best_val_loss_set += 1

        if self.epochs_since_best_val_loss_set < 2:
            self.linear_decay()

        self.curr_epoch += 1


    def linear_decay(self):
        linear_decay_ratio = self.initial_teacher_forcing_ratio * (1 - self.curr_epoch / self.max_epochs)
        self.curr_teacher_forcing_ratio = linear_decay_ratio


# In[ ]:


def plot_and_save(train_losses, val_losses, learning_rates):
    plt.style.use('classic')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Losses', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()
    ax.legend(fontsize=12)
    fig.savefig('losses.png')

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(learning_rates, label='Learning Rate')
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()
    ax.legend(fontsize=12)
    fig.savefig('learning_rates.png')


# In[ ]:


def save_lists_to_file(file_path, train_losses, val_losses, learning_rates):
    data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_lists_from_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['train_losses'], data['val_losses'], data['learning_rates']


# In[ ]:


# Needed when running inference on laptop
def adjust_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


# In[ ]:


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
tokenizer_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

train_dataset = CocoCaptions(root='./coco/images',
                       annFile='./coco/annotations/captions_train2014.json',
                       transform=image_transform)
val_dataset = CocoCaptions(root='./coco/images',
                           annFile='./coco/annotations/captions_val2014.json',
                           transform=image_transform)
train_captions = [entry['caption'] for entry in train_dataset.coco.anns.values()]
val_captions = [entry['caption'] for entry in val_dataset.coco.anns.values()]

caption_preprocessor = CaptionPreprocessor(train_captions + val_captions, tokenizer)

max_caption_length_train = max([len(caption.split()) for caption in train_captions])
max_caption_length_val = max([len(caption.split()) for caption in val_captions])
max_caption_length = max(max_caption_length_train, max_caption_length_val)

print(max_caption_length_train, max_caption_length_val, max_caption_length)
print('Maximum caption length (without <start>, <end>, and <pad> tokens):', max_caption_length)

average_caption_length = caption_preprocessor.get_average_caption_length()
std_dev_caption_length = caption_preprocessor.get_caption_length_standard_deviation()

print('Average caption length:', average_caption_length)
print('Standard deviation of caption length:', std_dev_caption_length)

custom_train_dataset = CustomCocoDataset(train_dataset, caption_preprocessor, num_captions=5)
custom_val_dataset = CustomCocoDataset(val_dataset, caption_preprocessor, num_captions=5)

batch_size = 320
# num_workers = os.cpu_count()
num_workers = 16
print('CPU COUNT:', num_workers)
train_data_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
val_data_loader = DataLoader(custom_val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


# In[ ]:


import numpy as np

def display_random_sample(dataset, tokenizer):
    random_idx = np.random.randint(0, len(dataset))
    image, caption = dataset[random_idx]
    image = image.permute(1, 2, 0).numpy()
    image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)

    caption_text = tokenizer.decode(caption.numpy(), skip_special_tokens=False)

    plt.imshow(image)
    plt.axis('off')
    plt.title(caption_text)
    plt.show()

# Display a random sample from the training dataset
# display_random_sample(custom_train_dataset, tokenizer)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_encoder = VisionTransformer(in_channels=3,
                                  patch_size=16,
                                  embed_dim=768,
                                  num_layers=4,
                                  num_heads=16,
                                  mlp_dim=768,
                                  num_classes=768).to(device)

auto_model = AutoModel.from_pretrained(tokenizer_name).to(device)
caption_decoder = TransformerCaptionDecoder(auto_model=auto_model,
                                            d_model=768,
                                            num_layers=4,
                                            num_heads=16,
                                            mlp_dim=768,
                                            dropout=0.2).to(device)

model = ImageCaptioningModel(image_encoder, caption_decoder).to(device)

useTwoGPUs = True
if torch.cuda.device_count() > 1 and useTwoGPUs:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)
# start newwwwwwww
num_epochs = 240

total_samples = len(train_data_loader.dataset)
batch_size = train_data_loader.batch_size
max_iterations = math.ceil(total_samples / batch_size)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0,) # weight_decay=2e-6)

warmup_steps = 1000
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.67, patience=2, verbose=True)
scheduler = NoamScheduler(optimizer, d_model=768, warmup_steps=warmup_steps, scaling_factor=0.01)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * max_iterations, eta_min=1e-6)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int((num_epochs * max_iterations) / 6.5), T_mult=2, eta_min=1e-6)

# tf scheduler
# tf_scheduler = TeacherForcingScheduler(num_epochs, warmup_steps=warmup_steps)

best_val_loss = float('inf')
val_loss = float('inf')

train_losses = []
val_losses = []
learning_rates = []

load_best_model = True
load_final = True
best_model_path = 'larger_attempt_3_FINAL_221_230_tf_0_6.pt'
save_lists_path = 'larger_attempt_3_FINAL_221_230_tf_0_6.pkl'
if load_best_model and os.path.exists(best_model_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
        adjusted_state_dict = adjust_state_dict_keys(checkpoint['model_state_dict'])
        model.load_state_dict(adjusted_state_dict)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.optimizer = optimizer
    scheduler.current_step = checkpoint['scheduler_state_dict']['current_step'] # use with noam
    # scheduler.state_dict = checkpoint['scheduler_state_dict'] # use with other schedulers
    # tf sched
    # tf_scheduler = checkpoint['tf_scheduler_state_dict']
    best_val_loss = checkpoint['best_val_loss']
    train_losses, val_losses, learning_rates = load_lists_from_file(save_lists_path)
    start_epoch = len(train_losses)
    if start_epoch >= num_epochs:
        raise ValueError('Number of epochs to train on is too small')
    training_range = range(start_epoch, num_epochs)
    print(len(train_losses))
    print('Loaded best saved model...')
    print(f'Validation loss of the loaded model: {best_val_loss:.4f}')
else:
    training_range = range(num_epochs)


# In[ ]:


new_dropout = True
new_dropout_rate = 0.2
if new_dropout:
    if isinstance(model, nn.DataParallel):
        model.module.update_dropout_rate(new_dropout_rate)
    else:
        model.update_dropout_rate(new_dropout_rate)


# In[ ]:


# stepCounter = stepCounter() # use with other schedulers


# In[ ]:


print('**********STARTING TRAINING**********')
training_start = time.time()
for epoch in training_range:
    epoch_start = time.time()

    print(f'Total samples: {total_samples}, Batch size: {batch_size}, Maximum iterations: {max_iterations}')

    avg_every = 50
    old_lr = optimizer.param_groups[0]['lr']
    # print(old_lr, stepCounter.steps) # use with other schedulers
    print(old_lr, scheduler.current_step) # use with noam

    # tf scheduler
    # old_tf_ratio = tf_scheduler.curr_teacher_forcing_ratio
    # tf_scheduler.step(val_loss, scheduler.current_step)

    train_loss = train_one_epoch(model, train_data_loader, criterion, optimizer, scheduler, device, epoch, num_epochs, avg_every, learning_rates, teacher_forcing_ratio=1.0) # stepCounter) # use with other schedulers
    # 1-100: 1.0
    # 101-120: 0.9
    # 121-160: 0.8
    # 161-170: 0.9
    # 171-220: 0.7
    # 221-230: 0.6
    # 231-240: 1.0, decoder dropout: 0.2
    print(f'TRAINING LOSS FOR EPOCH {epoch + 1}: {train_loss:.4f}')

    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f'****LR changed from {old_lr} ==> {new_lr}****')

    val_loss = evaluate(model, val_data_loader, criterion, device)
    print(f'CURRENT BEST VALIDATION LOSS: {best_val_loss:.4f}')
    print(f'VALIDATION LOSS FOR EPOCH {epoch + 1}: {val_loss:.4f}')

    # tf scheduler
    # new_tf_ratio = tf_scheduler.curr_teacher_forcing_ratio
    # if old_tf_ratio != new_tf_ratio:
    #     print(f'****TF RATIO CHANGED FROM {old_tf_ratio} ==> {new_tf_ratio}****')

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        if load_final:
            save_name = 'larger_attempt_3.pt'
            save_lists_path = 'larger_attempt_3.pkl'
        else:
            save_name = best_model_path

        best_val_loss = val_loss

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.__dict__, # use with noam
            # 'scheduler_state_dict': scheduler.state_dict(), # use with other schedulers
            'best_val_loss': best_val_loss,
            # 'tf_scheduler_state_dict': tf_scheduler.__dict__
        }, save_name)
        save_lists_to_file(save_lists_path, train_losses, val_losses, learning_rates)
        print(f'********************NEW BEST MODEL SAVED @ VAL: {best_val_loss:.4f}********************')

    if epoch == num_epochs - 1:
        final_val_loss = best_val_loss
        final_save_name = 'larger_attempt_3_FINAL_231_240_tf_1_0_dropOut_0_2.pt'
        final_save_lists = 'larger_attempt_3_FINAL_231_240_tf_1_0_dropOut_0_2.pkl'

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.__dict__, # use with noam
            # 'scheduler_state_dict': scheduler.state_dict(), # use with other schedulers
            'best_val_loss': final_val_loss,
            # 'tf_scheduler_state_dict': tf_scheduler.__dict__
        }, final_save_name)
        save_lists_to_file(final_save_lists, train_losses, val_losses, learning_rates)

    epoch_end = time.time()
    print(f'Epoch {epoch+1}/{num_epochs} total time: {epoch_end - epoch_start}')

training_end = time.time()
print(f'Total training time: {training_end - training_start}')



plot_and_save(train_losses, val_losses, learning_rates)


# In[ ]:


# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#
# test_dataset = CocoCaptions(root='./coco/images',
#                             annFile='./coco/annotations/image_info_test2014.json',
#                             transform=image_transform)
# custom_test_dataset = CustomCocoDataset(test_dataset, caption_preprocessor, num_captions=5)


# In[ ]:


# def generate_caption(model, image, tokenizer, device, max_len=14):
#     model.eval()
#     with torch.no_grad():
#         image = image.unsqueeze(0).to(device)
#         image_features = model.image_encoder(image)
#         start_token_tensor = torch.tensor([model.start_token_index], dtype=torch.long, device=device)
#         start_token_embeddings = model.caption_decoder.auto_model.embeddings(start_token_tensor).repeat(image_features.shape[0], 1, 1)
#         image_features_summed = image_features.sum(dim=1).unsqueeze(1)
#         image_features_summed = model.image_feature_linear(image_features_summed)
#         memory = torch.cat([start_token_embeddings, image_features_summed], dim=1)
#         memory = memory.transpose(0, 1)
#
#         captions_output = torch.zeros((1, max_len)).long().to(device)
#         captions_output[:, 0] = model.start_token_index
#
#         for t in range(1, max_len):
#             captions_input = captions_output[:, :t].to(device)
#             output = model.caption_decoder(captions_input, memory[:, :t].clone())
#             captions_output[:, t] = output[:, -1].argmax(-1)
#
#         decoded_caption = tokenizer.decode(captions_output.squeeze().tolist(), skip_special_tokens=False)
#
#     return decoded_caption
#
#
# def display_image(image, caption_text):
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(caption_text)
#     plt.show()
#
#
# inverse_transform = Compose([
#     Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
# ])
#
# random_idx = random.randint(0, len(custom_test_dataset) - 1)
# transformed_image, _ = custom_test_dataset[random_idx]
# generated_caption = generate_caption(model, transformed_image, tokenizer, device)
#
# original_image = inverse_transform(transformed_image).permute(1, 2, 0).numpy()
# original_image = np.clip(original_image, 0, 1)
#
# print('Generated caption:', generated_caption)
# display_image(original_image, generated_caption)

