import torch
from torch import nn
from torch.nn import functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel 
import numpy as np
from sklearn.preprocessing import StandardScaler


class DDPM(nn.Module):
    def __init__(self, args, aconc=None, weather=None):
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
        self.net.load_state_dict(torch.load(args.checkpoint_model_dir))
        self.net.to(args.device)
        self.loss_fn = nn.MSELoss()
        self.channels = args.channels
        self.layers = args.layers
        self.batch_size = args.batch_size
        self.device = args.device
        self.ddimsteps = args.ddimsteps
        if args.mode == 'train':
            self.aconc = aconc
            self.weather = weather
            self.R2 = args.R2[:args.multistep]
            self.weights = self.W(self.R2)

    def W(self, R2):
        R2_normalized = (R2 - min(R2)) / (max(R2) - min(R2))
        weights = 1 - R2_normalized 
        weights /= weights.sum()
        return weights

    def forward(self, x, batch_count, multistep):
        loss_batch = None
        for step in range(multistep):    
            x = x.to(self.device).to(torch.float32)
            start = batch_count * self.batch_size + step
            end = start + self.batch_size
            x_t = torch.cat((x,self.weather[start:end,:]),dim=1) # all variants at t timepoint of atmospheric process
            x_tt_aconc = self.aconc[start+1:end+1,:]             # ACONC at t+1 timepoint of atmospheric process
            noise = x                                            # noise, defined as ACONC at t timepoint
            noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear', timestep_spacing="trailing")
            timesteps = torch.linspace(0,999, self.batch_size).long()
            noised_xtt = noise_scheduler.add_noise(x_tt_aconc, noise, timesteps) # noised ACONC at t+1 timepoint
            x_t[:,:self.layers,:,:] = noised_xtt
            pred = self.net(x_t, timesteps)['sample']                            # predicted noise
            if loss_batch == None:
                loss_batch = self.weights[step]*self.loss_fn(pred, noise) 
            else: 
                loss_batch += self.weights[step]*self.loss_fn(pred, noise)
            # Sampling 
            x_t = self.aconc[start:end,:]
            noise_scheduler.set_timesteps(num_inference_steps=self.ddimsteps)    # Sampling steps of DDIM
            for t in noise_scheduler.timesteps:
                t_tensor = torch.tensor([t], dtype=torch.long).to(self.device)
                all_var = torch.cat((x_t, self.weather[start:end,:]), dim=1)
                pred_noise = self.net(all_var, t_tensor)['sample']  
                step_result = noise_scheduler.step(
                    model_output=pred_noise,           
                    timestep=t_tensor,                 
                    sample=all_var[:,:self.layers,:,:] 
                )
                x_t = step_result.prev_sample
            x = x_t 
            print(f'Done for {step+1} step!')   
        return loss_batch
    
    def Reverse(self, data, first):
        sample = None
        x = first.to(self.device)
        x_t = x[:,:self.layers,:,:] 
        with torch.no_grad():
            for i in range(24):
                noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear', timestep_spacing="trailing")
                noise_scheduler.set_timesteps(num_inference_steps=self.ddimsteps)
                weather = data[i,self.layers:self.channels,:,:].unsqueeze(0)
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
                    if t == noise_scheduler.num_train_timesteps-1 :
                        if sample is None:
                            sample = x_t.detach().numpy()
                        else:
                            sample = np.concatenate((sample, x_t.detach().numpy()), axis=0)
                    del pred_noise, step_result, all_var
                    torch.cuda.empty_cache()
            print(f'Done for Step {i+1}!')
        return sample

if __name__ == '__main__':
    pass

