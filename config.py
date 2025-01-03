from dataclasses import dataclass

@dataclass
class Config():
    seed = 5401

    au2lm="weights/au2lm/CREMA/checkpoints/weight.ckpt"
    lm2face="weights/lm2face/CREMA"

    num_inference_steps=8

    dset="CREMA"
    noise_ksize=25 #11 with MEAD 25 with CREMA
    audio="temp/data/au/sample/1015_DFA_FEA_XX.wav"
    audio= None # audio path
    ident= None # identity path

    output= None # output path

    # emotions = {0:'angry', 1:'fear', 2:'sad', 3:'contempt', 4:'happy', 5:'surprised', 6:'disgusted', 7:'neutral'}