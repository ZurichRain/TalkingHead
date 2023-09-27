import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
import torch.nn as nn
import torch
from transformers import CLIPVisionModel

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Juliet(nn.Module):
    def __init__(self,
                 noise_scheduler,
                 unet,
                 model_args, 
                 vae = None,
                 text_encoder=None,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.unet = unet
        self.vae = vae

        if model_args.model.freeze_vae:
            self.vae.requires_grad_(False)

        # if model_args.model
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.args = model_args
        self.img_adapter = MLP(1024, 1024 , 768).apply(init_weights)
        self.vision_tower = CLIPVisionModel.from_pretrained('/data/public/lucaszhao/data/llm/clip/vit-large-patch14')

        if model_args.model.freeze_vision_tower:
            self.vision_tower.requires_grad_(False)

        # torch.nn.init.xavier_uniform_(self.img_adapter.weight)



    def forward(self, batch):

        latents = self.vae.encode(batch["cur_frame_image"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.args.model.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.args.model.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if self.args.model.input_perturbation:
            new_noise = noise + self.args.model.input_perturbation * torch.randn_like(noise)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.args.model.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # print(batch["cur_frame_image"].size())
        # cur_f_img = batch["cur_frame_image"].view(bsz, batch["cur_frame_image"].size()[1], -1)
        # print(cur_f_img.size())
        # encoder_hidden_states = self.vae.encode(batch["last_frame_image"]).latent_dist[0]
        # self.vision_tower()
        # print(batch["last_frame_image"].size())
        image_forward_out = self.vision_tower(batch["last_frame_image"], output_hidden_states=True)
        select_hidden_state = image_forward_out.hidden_states[-1]
        image_feature = select_hidden_state[:, 1:]
        # print(image_feature.size())
        # torch.Size([1, 3, 224, 224])
        # self.vision_tower.config.hidden_size

        encoder_hidden_states = image_feature.view(bsz, image_feature.size()[1], -1)
        # .view(bsz, batch["cur_frame_image"].size()[1], -1)
        # print(encoder_hidden_states.size())
        encoder_hidden_states = self.img_adapter(encoder_hidden_states)
        # encoder_hidden_states = self.vae.encode(cur_f_img).latent_dist.sample()

        # Get the target for loss depending on the prediction type
        if self.args.model.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.args.model.prediction_type)

        # if self.noise_scheduler.config.prediction_type == "epsilon":
        #     target = noise
        # elif self.noise_scheduler.config.prediction_type == "v_prediction":
        #     target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        # else:
        #     raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return model_pred, noise , latents, timesteps