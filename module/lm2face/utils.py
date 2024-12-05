################################################################
import torch
from torchvision.transforms.functional import gaussian_blur, normalize


def recvt(tensor, mx, mn):
    return (tensor*(mx - mn) + mn)/2

def cvt(tensor, mx, mn):
    return (tensor - mn) / (mx - mn)

def add_noise_image(img, lmd, lm_min, lm_max, ksize):
    x = torch.stack([img for i in range(len(lmd))])
    lm = recvt(lmd, lm_max,lm_min).type(torch.int32)
    mask = torch.zeros(x.shape[:3]).to(img.device)
    for idx in range(mask.shape[0]):
        mask[idx][lm[idx,1,:], lm[idx,0,:]] = 1
    mask = gaussian_blur(mask, kernel_size=ksize)
    mask = torch.stack([mask, mask, mask], dim=-1)
    mask = cvt(mask, mask.max(), mask.min())
    mask += 0.1

    noise = mask * torch.randn(x.shape).to(img.device)
    return (x + noise).permute(3,1,2,0), x.permute(3,1,2,0)

def add_noise_batch(ident, lmd, mn, mx, ksize):
    mask_noise = []
    images = []
    for sf, lmd in zip(ident, lmd):
        mk, img = add_noise_image(sf, lmd, mn, mx, ksize)
        mask_noise.append(mk)
        images.append(img)
    mask_noise = torch.stack(mask_noise)
    images = torch.stack(images)
    return mask_noise, images