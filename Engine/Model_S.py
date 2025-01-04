import torch
from torch import nn
from torch.nn import functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel 
import numpy as np

class DDPM(nn.Module):
    def __init__(self, args):
        super(DDPM, self).__init__()
        self.net = UNet2DModel(
                                sample_size=128,
                                in_channels=args.channels,
                                out_channels=args.layers,
                                layers_per_block=2,
                                block_out_channels=(64,64,128), 
                                down_block_types=(
                                                    'DownBlock2D',
                                                    'AttnDownBlock2D',
                                                    'AttnDownBlock2D',
                                                 ),
                                up_block_types=(
                                                    'AttnUpBlock2D',
                                                    'AttnUpBlock2D',
                                                    'UpBlock2D',
                                               ),
                               )
        if args.mode == 'sampling':
            self.net.load_state_dict(torch.load(args.checkpoint_model_dir))
        if args.mode == 'train' and args.pmode == 'multi':
            self.net.load_state_dict(torch.load(args.checkpoint_model_dir))
        self.net.to(args.device)
        self.channels = args.channels
        self.batch_size = args.batch_size
        self.device = args.device
        self.layers = args.layers
        self.ddimsteps = args.ddimsteps

    def forward(self, x):
        x = x.to(self.device).to(torch.float32)
        x_t = x[:,:self.channels,:]        
        x_tt_aconc = x[:,self.channels:,:] 
        noise = x_t[:,:self.layers,:]
        noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear')
        timesteps = torch.linspace(0,999, self.batch_size).long().to(self.device)
        noised_xtt = noise_scheduler.add_noise(x_tt_aconc, noise, timesteps)
        x_t[:,:self.layers,:,:] = noised_xtt
        pred = self.net(x_t, timesteps)['sample'] 
        return pred, noise
    
    def Reverse_I(self, x, real, sample):
        x = x.to(self.device).to(torch.float32)
        x_t = x[:,:self.channels,:,:]
        x_tt = x[:,self.channels:,:,:]
        if real is None:
            real = x_tt.numpy()
        else:
            real = np.concatenate((real, x_tt.numpy()), axis=0)
        noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear', timestep_spacing="trailing") 
        noise_scheduler.set_timesteps(num_inference_steps=self.ddimsteps)
        with torch.no_grad():
            weather = x_t[:,self.layers:,:,:]
            x_t = x_t[:,:self.layers,:,:]
            for t in noise_scheduler.timesteps:
                t_tensor = torch.tensor([t], dtype=torch.long).to(self.device)
                all_var = torch.cat((x_t, weather), dim=1)
                pred_noise = self.net(all_var, t_tensor)['sample']  
                step_result = noise_scheduler.step(
                    model_output=pred_noise,  
                    timestep=t_tensor,        
                    sample=all_var[:,:self.layers,:,:] 
                )
                x_t = step_result.prev_sample.detach() 
                del pred_noise, step_result, all_var
                torch.cuda.empty_cache()
                print(f'Done for Denoising timestep {t}!')
            if sample is None:
                sample = x_t.detach().numpy()
            else:
                sample = np.concatenate((sample, x_t.detach().numpy()), axis=0)
        return real, sample
        
if __name__ == '__main__':
    pass

