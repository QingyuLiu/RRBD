import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler

def ddim_loop_target_step(pipe,prompt,latents,target_step,device="cuda:0",num_inference_steps=50, guidance_scale = 1):
    prompt_embeds,negative_prompt_embeds  = pipe.encode_prompt(prompt, device, 1, True, None)
    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
    # latents *= pipe.scheduler.init_noise_sigma


    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # noises = []
    # latents_list=[latents]

    timesteps = []
    x_0_forward_noise =[]
    # inversion process
    for i, t in enumerate(pipe.scheduler.timesteps):

        if i > target_step:
            break
        timesteps.append(t)
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        x_0_forward_noise.append(noise_pred)
        # noises.append(noise_pred)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        # latents_list.append(latents)
    # x_0_forward_noise = noise_pred
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # sampling process
    x_1_backward_noise=[]
    for i, t in enumerate(reversed(timesteps)):
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        x_1_backward_noise.append(noise_pred)
        # noises.append(noise_pred)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        # latents_list.append(latents)
    theoretical_residual = latents * 0
    for i, t in enumerate(timesteps):
        alpha_prod_i = pipe.scheduler.alphas_cumprod[t]
        alpha_i = pipe.scheduler.alphas[t]
        numerator =  (1 - alpha_prod_i) ** 0.5 -(alpha_i - alpha_prod_i) ** 0.5
        denominator = alpha_prod_i ** 0.5
        theoretical_residual  += (numerator / denominator) * (x_1_backward_noise[len(timesteps)-i-1]-x_0_forward_noise[i])
    return latents,theoretical_residual