import torch
import pipeline
import model_loader
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
import torchaudio
import torch.nn.functional as F
from diffusion import Diffusion
from encoder import VAE_Encoder
from decoder import VAE_Decoder
import pandas as pd

from dataclasses import dataclass

AUDIO_PATH = "../audio/"
LABELS_FILE_PATH = AUDIO_PATH + "labels.csv"

SAMPLE_RATE = 44100
INPUT_AUDIO_LENGTH_SECONDS = 1


@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

# load dataset

dataset = torch.randn(2, 44100, 0)

desired_dimensions = [2, 44100]

data = pd.read_csv(LABELS_FILE_PATH, names=["id", "label", "audio_path"])

for path in data["audio_path"]:
    waveform, sample_rate = torchaudio.load(AUDIO_PATH + path)

    if waveform.size(1) < INPUT_AUDIO_LENGTH_SECONDS * SAMPLE_RATE:
        padding_width = desired_dimensions[1] - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding_width), mode='constant', value=0)
    elif waveform.size(1) > INPUT_AUDIO_LENGTH_SECONDS * SAMPLE_RATE:
        waveform = waveform[:, :desired_dimensions[1]]
    

    waveform = waveform.unsqueeze(2)
    dataset = torch.cat([dataset, waveform], dim=2)


# preprocess data

print(dataset[:2, :5, :5])

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)



encoder = VAE_Encoder()
diffusion = Diffusion()
decoder = VAE_Decoder()

loss_function = torch.nn.MSELoss()  # Choose an appropriate loss function
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(diffusion.parameters()) + list(decoder.parameters()), lr=config.learning_rate)

noise_scheduler = DDPMSampler()


# Training loop
for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_dataloader))
    progress_bar.set_description(f"Epoch {epoch}")

    for batch_data in train_dataloader:
        # Forward pass
        input_data = batch_data  # Adjust as needed, make sure these are just the audios/images

        # Generate noise
        noise = torch.randn(input_data.shape, device=input_data.device)
        bs = input_data.shape[0]

        timesteps = torch.randint(
                0, noise_scheduler.timesteps, (bs,), device=input_data.device,
                dtype=torch.int64
            )
        
        #noisy_images = noise_scheduler.add_noise(input_data, noise, timesteps)

        latents = encoder(input_data, noise)
        diffused_output = diffusion(latents, timesteps)
        output = decoder(diffused_output)

        # Compute loss
        loss = loss_function(output, noise)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print epoch loss or other metrics
    print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item()}')

torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'diffusion_state_dict': diffusion.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, '../data/data.pth')
