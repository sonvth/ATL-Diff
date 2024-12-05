from module.au2lm.KFLM import KFLM
from module.lm2face.diffusion.pipeline import MyPipeline
from module.lm2face.utils import *
import torchaudio
import torchvision
import face_alignment
import numpy as np
import matplotlib.pyplot as plt
import cv2

from config import Config

import warnings
warnings.filterwarnings('ignore')


def load_au2lm(config):
    model = KFLM.load_from_checkpoint(config.au2lm, batch=1, init_lr=1e-3, num_of_landmarks=68)
    model.to("cuda")
    return model

# def load_lm2face(config):
#     pipeline = MyPipeline.from_pretrained(config.lm2face, use_safetensors=True)
#     pipeline=pipeline.to("cuda")
#     print(pipeline)
#     return pipeline


def load_input(config, au_path, ident_path):
    x, _ = torchaudio.load(f"{au_path}", normalize="True")
    if config.dset == "MEAD":
        x = torchaudio.transforms.Resample(48000, 16000, dtype=x.dtype)(x)

    x = torch.unsqueeze(torch.mean(x, dim=0),0)
    if "mp4" in ident_path:
        ident = torchvision.io.read_video(f"{ident_path}", pts_unit="sec",output_format="THWC")[0]
        ident = ident[:1]

    else:
        ident = torchvision.io.read_image(f"{ident_path}")
        ident = ident.permute(1,2,0)
        ident = ident.unsqueeze(dim=0)
    emo = torch.tensor([config.emotion])
    x = x.to('cuda')
    w = emo.to('cuda')

    return x, ident, w

def extract_lm(config, au2lm, x, ident, w):

    align = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')
    ilm = torch.tensor(np.array(align.get_landmarks(ident))).permute(0,2,1)
    ilm /= ident.shape[0]
    v = ilm.contiguous().to('cuda')

    y_ = au2lm(x,v,w)
    y_new = torch.cat((y_, y_[:,-2:,:,:]), dim=1)
    return y_new

def load_model(config):

    pipeline = MyPipeline.from_pretrained(config.lm2face, use_safetensors=True)
    au2lm = load_au2lm(config)

    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False

    au2lm = au2lm.to('cuda')
    pipeline=pipeline.to('cuda')
    resnet=resnet.to('cuda')

    return au2lm, pipeline, resnet


def gen_(config, pipeline, resnet, noise, ident, labels):
    with torch.no_grad():
        ident = resnet(ident.permute(0,3,1,2).to('cuda'))
        ident = ident.reshape([1, 64, 32])

    noise = noise.to('cuda')
    labels = labels.to('cuda')
    pred,_ = pipeline(noise,ident,labels,num_inference_steps=config.num_inference_steps)
    return pred


from module.lm2face.utils import *
import torchvision
from config import Config
import warnings
warnings.filterwarnings('ignore')



def main(config):
    torch.manual_seed(5401)

    x, video, w = load_input(config, config.audio, config.ident)

    # Length of each segment (e.g., 1 second = 16000 samples)
    segment_length = 16000
    num_segments = x.shape[1] // segment_length

    # List to store video frames
    video_segments = []
 
    # Load models (audio-to-landmark, pipeline, and resnet)
    au2lm, pipeline, resnet = load_model(config)

    # Use the first frame of the video as the initial identity (ident)
    ident = torchvision.transforms.Resize((128,128))(video.permute(0, 3, 1, 2))
    ident = ident.permute(0, 2, 3, 1).float() / 255

    lmd = []
    for i in range(num_segments):

        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        x_segment = x[:, start_idx:end_idx]
        lmd.append(extract_lm(config, au2lm, x_segment, video[0], w)[0])
    
    lmd = torch.cat(lmd, dim=0)
    noises, _ = add_noise_batch(ident, lmd.unsqueeze(dim=0), 0, 255, ksize=config.noise_ksize)
    noises = torch.split(noises, 32, dim=-1)

    for noise in noises:
        pred = gen_(config, pipeline, resnet, noise, ident, w)
        pred = pred[0].permute(3, 1, 2, 0)
        pred = pred.clamp(0, 1)
        video_segments.append(pred.cpu() * 255)

        out = torch.concatenate(video_segments,dim=0)
        out = out.reshape(-1,32,128,128,3)
        out = out[:, 5:29, :, :, :]

    for idx in range(len(out) - 1):
            last_frames = out[idx] 
            next_frames = out[idx + 1]

            temp_last = last_frames.clone().numpy()
            temp_next = next_frames.clone().numpy()
            
            for f_idx in range(8):
                a = 1/(f_idx + 1)
                
                temp_last[-f_idx-1] = crossfade(last_frames[-f_idx-1], next_frames[3], a)

            out[idx] = torch.tensor(temp_last)
            out[idx+1] = torch.tensor(temp_next) 


    resemb = out.reshape(-1,128,128,3)

    torchvision.io.write_video(
        f"{config.output}_.mp4",
        resemb,
        fps=24,
        audio_array=x.cpu(),  # Adjust audio length
        audio_fps=16000,
        audio_codec='aac'
    )


# def main(config):
#     # torch.manual_seed(5401)

#     x, video, w = load_input(config, config.audio, config.ident)

#     # Length of each segment (e.g., 1 second = 16000 samples)
#     segment_length = 16000
#     num_segments = x.shape[1] // segment_length

#     # List to store video frames
#     video_segments = []
 
#     # Load models (audio-to-landmark, pipeline, and resnet)
#     au2lm, pipeline, resnet = load_model(config)

#     # Use the first frame of the video as the initial identity (ident)
#     ident = torchvision.transforms.Resize((128,128))(video.permute(0, 3, 1, 2))
#     ident = ident.permute(0, 2, 3, 1).float() / 255

#     for i in range(num_segments):
#         # print(ident[:,:,:,0].min(), ident[:,:,:,0].max())
#         # print(ident[:,:,:,1].min(), ident[:,:,:,1].max())
#         # print(ident[:,:,:,2].min(), ident[:,:,:,2].max())


#         start_idx = i * segment_length
#         # if i > 0:
#         #     start_idx -= 5000
#         end_idx = start_idx + segment_length
#         x_segment = x[:, start_idx:end_idx]

#         lmd = extract_lm(config, au2lm, x_segment, video[0], w)
#         noise, _ = add_noise_batch(ident, lmd, 0, 255, ksize=config.noise_ksize)
#         pred = gen_(config, pipeline, resnet, noise, ident, w)
#         pred = pred[0].permute(3, 1, 2, 0)

#         # ident = abs(ident.to('cuda')*0.7 + pred[-1:].clone() * 0.3)
#         # ident = ident.clamp(0, 1)

#         pred = pred.clamp(0, 1)
#         video_segments.append(pred.cpu() * 255)

#         # pred[:, 0, :, :] = pred[:, 0, :, :].clamp(0, 1)  
#         # pred[:, 1, :, :] = pred[:, 1, :, :].clamp(0, 0.8)
#         # pred[:, 2, :, :] = pred[:, 2, :, :].clamp(0, 0.7)

#     out = torch.concatenate(video_segments,dim=0)
#     out = out.reshape(-1,32,128,128,3)
#     out = out[:, 4:28, :, :, :]

#     for idx in range(len(out) - 1):
#         # if (idx + 1) % 3 == 0: 
#             last_frames = out[idx] 
#             next_frames = out[idx + 1]

#             temp_last = last_frames.clone().numpy()
#             temp_next = next_frames.clone().numpy()
            
#             for f_idx in range(10):
#                 a = 1/(f_idx + 1)
                
#                 temp_last[-f_idx-1] = crossfade(last_frames[-f_idx-1], next_frames[3], a)
#                 # temp_next[f_idx] = crossfade(next_frames[f_idx], next_frames[3], 1 - a)
#                 # temp_last[-f_idx-1] = 0
#                 # temp_next[f_idx] = 0

#             out[idx] = torch.tensor(temp_last)
#             out[idx+1] = torch.tensor(temp_next) 


#     resemb = out.reshape(-1,128,128,3)

#     torchvision.io.write_video(
#         f"{config.output}_.mp4",
#         resemb,
#         fps=24,
#         audio_array=x.cpu(),  # Adjust audio length
#         audio_fps=16000,
#         audio_codec='aac'
#     )
 
def crossfade(frame1, frame2, alpha):
    return (1 - alpha) * frame1 + alpha * frame2

if __name__ == "__main__":
    config = Config()
    main(config)

