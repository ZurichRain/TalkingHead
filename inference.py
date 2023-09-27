from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

# model_path = "/data/public/sharpwang/genHandImg/diffusers/examples/text_to_image/latex2math-lora/checkpoint-10000"

# unet = UNet2DConditionModel.from_pretrained(model_path + "/unet")
# # model_path = "sayakpaul/sd-model-finetuned-lora-t4"
# # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
# # pipe.unet.load_attn_procs(model_path)
# # pipe.to("cuda")


# # pipe = StableDiffusionPipeline.from_pretrained("/data/public/sharpwang/llm/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.bfloat16)
# pipe = StableDiffusionPipeline.from_pretrained("/data/public/sharpwang/llm/stable-diffusion-v1-4", unet=unet)
# pipe.to("cuda")

# image = pipe(prompt="").images[0]
# image.save("1.png")


from models.Juliet import Juliet
from utils.utils import get_pretrained_model
from utils.get_parser import get_parser
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms
from PIL import Image
import inspect

args = get_parser()
model = get_pretrained_model(args)

model.load_state_dict(torch.load("/data/hypertext/sharpwang/TalkingHead/MyCode/checkpoint_dir/Jack_Ma+Kathleen+Theresa_May+Emmanuel_Macron+Donald_Trump/checkpoint-3600"))


# 0. Default height and width to unet
# height = height or model.unet.config.sample_size * model.vae_scale_factor
# width = width or model.unet.config.sample_size * model.vae_scale_factor

# 1. Check inputs. Raise error if not correct
# model.check_inputs(
#     prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
# )

img_path ="/data/hypertext/sharpwang/TalkingHead/mahuat.jpg"
height = 512
width = 512
num_inference_steps = 50
eta = 0.0

generator = None

train_transforms = transforms.Compose(
        [
            transforms.Resize(args.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.data.resolution) if args.data.center_crop else transforms.RandomCrop(args.data.resolution),
            transforms.RandomHorizontalFlip() if args.data.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

train_transforms_last = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224) if args.data.center_crop else transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip() if args.data.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def prepare_extra_step_kwargs(model, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(model.noise_scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(model.noise_scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

def get_last_frame_image():
    
    # default to score-sde preprocessing
    image = Image.open(img_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    # examples["pixel_values"] = [train_transforms(image) for image in images]
    image = train_transforms_last(image)
    # image = np.array(image).astype(np.uint8)
    # image = Image.fromarray(image)
    # h, w = image.size
    # if self.size is not None:
    #     image = image.resize((self.size, self.size), resample=self.interpolation)

    # image = np.array(image).astype(np.uint8)
    # image = (image / 127.5 - 1.0).astype(np.float32)
    return image

# def get_img_last(self, image_path):
        

last_frame_image = get_last_frame_image()
last_frame_image = torch.stack([last_frame_image])
last_frame_image = last_frame_image.to(memory_format=torch.contiguous_format).float()

batch_size = 1



# 2. Define call parameters
# if prompt is not None and isinstance(prompt, str):
#     batch_size = 1
# elif prompt is not None and isinstance(prompt, list):
#     batch_size = len(prompt)
args.train.device = "cuda" if torch.cuda.is_available() else "cpu"
device = args.train.device
model = model.to(device)
last_frame_image = last_frame_image.to(device)
model=model.eval()
# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
# corresponds to doing no classifier free guidance.
# do_classifier_free_guidance = guidance_scale > 1.0


vae_scale_factor = 2 ** (len(model.vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)


# 3. Encode input prompt
# text_encoder_lora_scale = (
#     cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
# )
# prompt_embeds = self._encode_prompt(
#     prompt,
#     device,
#     num_images_per_prompt,
#     do_classifier_free_guidance,
#     negative_prompt,
#     prompt_embeds=prompt_embeds,
#     negative_prompt_embeds=negative_prompt_embeds,
#     lora_scale=text_encoder_lora_scale,
# )
# prompt_embeds = 

# 4. Prepare timesteps
model.noise_scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = model.noise_scheduler.timesteps

# 5. Prepare latent variables
num_channels_latents = model.unet.config.in_channels
# latents = self.prepare_latents(
#     batch_size * num_images_per_prompt,
#     num_channels_latents,
#     height,
#     width,
#     prompt_embeds.dtype,
#     device,
#     generator,
#     latents,
# )



shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
latents = randn_tensor(shape, generator=generator, device=device, dtype=last_frame_image.dtype)

# 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
extra_step_kwargs = prepare_extra_step_kwargs(model, generator, eta)

# 7. Denoising loop
num_warmup_steps = len(timesteps) - num_inference_steps * model.noise_scheduler.order
# with self.progress_bar(total=num_inference_steps) as progress_bar:

with torch.no_grad():
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = latents
        latent_model_input = model.noise_scheduler.scale_model_input(latent_model_input, t)

        
        image_forward_out = model.vision_tower(last_frame_image, output_hidden_states=True)
        select_hidden_state = image_forward_out.hidden_states[-1]
        image_feature = select_hidden_state[:, 1:]
        # print(image_feature.size())
        # torch.Size([1, 3, 224, 224])
        # self.vision_tower.config.hidden_size

        encoder_hidden_states = image_feature.view(batch_size, image_feature.size()[1], -1)
        # .view(bsz, batch["cur_frame_image"].size()[1], -1)
        # print(encoder_hidden_states.size())
        encoder_hidden_states = model.img_adapter(encoder_hidden_states)

        # encoder_hidden_states= model.vae.encode(last_frame_image).latent_dist.sample()
        # encoder_hidden_states = encoder_hidden_states.view(1, encoder_hidden_states.size()[1], -1)
        # encoder_hidden_states = model.img_adapter(encoder_hidden_states)
        # predict the noise residual
        noise_pred = model.unet(latent_model_input, t, encoder_hidden_states).sample
        # noise_pred = model.unet(
        #     latent_model_input,
        #     t,
        #     encoder_hidden_states=prompt_embeds,
        #     cross_attention_kwargs=cross_attention_kwargs,
        #     return_dict=False,
        # )[0]

        # perform guidance
        # if do_classifier_free_guidance:
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # if do_classifier_free_guidance and guidance_rescale > 0.0:
        #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        latents = model.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # # call the callback, if provided
        # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
        #     progress_bar.update()
        #     if callback is not None and i % callback_steps == 0:
        #         callback(i, t, latents)

image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
# image = self.pt_to_numpy(image)
images = image.cpu().permute(0, 2, 3, 1).float().numpy()
# image = self.numpy_to_pil(image)

# if images.ndim == 3:
#     images = images[None, ...]
# images = (images * 255).round().astype("uint8")
# if images.shape[-1] == 1:
#     # special case for grayscale (single channel) images
#     pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
# else:
#     pil_images = [Image.fromarray(image) for image in images]
# if has_nsfw_concept is None:
#     do_denormalize = [True] * image.shape[0]
# else:
#     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

do_denormalize = [True] * image.shape[0]
output_type = "pil"
image = image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

image[0].save("1.png")

# Offload last model to CPU
# if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
#     self.final_offload_hook.offload()

# if not return_dict:
#     return (image, has_nsfw_concept)