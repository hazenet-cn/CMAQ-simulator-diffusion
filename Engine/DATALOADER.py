import torch
import Engine.DATASET_Single as DDS
import Engine.DATASET_Multi as DDM
from torch.utils.data.distributed import DistributedSampler

def build_dataloader_S(args):
    if args.pmode != 'single':
        raise ValueError("The model must be Single Step!")
    batch_size = args.batch_size
    DATASET = DDS.CustomDataset(args)
    dataset = DATASET.data
    if args.mode == 'train': 
        jud =True 
    else: 
        jud = False
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=jud,
                                             num_workers=0,
                                             sampler=None
                                             )
    return dataloader, DATASET.scaler

def build_dataloader_M(args):
    if args.pmode != 'multi':
        raise ValueError("The model must be Multi Step!")
    DATASET = DDM.CustomDataset(args)
    if args.mode == 'train':
        aconc = DATASET.aconc
        weather = DATASET.weather
        dataloader = torch.utils.data.DataLoader(aconc,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=0,
                                            sampler=None
                                                )
        return dataloader, aconc, weather
    if args.mode == 'sampling':
        first = DATASET.first
        real = DATASET.real
        data = DATASET.data
        scaler = DATASET.scaler
        return first,real,data,scaler
    
def UnNorm(sample, real, scaler):
    a,b,c,d = sample.shape
    sample = sample.reshape(a,b*c*d)
    sample = scaler.inverse_transform(sample)
    sample = sample.reshape(a,b,c,d)
    a,b,c,d = real.shape
    real = real.reshape(a,b*c*d)
    real = scaler.inverse_transform(real)
    real = real.reshape(a,b,c,d)
    return sample, real

if __name__ == '__main__':
    pass
