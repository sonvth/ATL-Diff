import torch
from diffusers import DiffusionPipeline, DDIMScheduler


class MyPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, noise, ident, labels, num_inference_steps: int = 50):
        noise = noise.to(self.device)
        ident = ident.to(self.device)
        self.pred = []
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(noise, t, ident, class_labels=labels).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            noise = self.scheduler.step(model_output, t, noise).prev_sample
            self.pred.append(noise.permute(0,4,2,3,1)[0][10].cpu())
        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()

        return noise, model_output