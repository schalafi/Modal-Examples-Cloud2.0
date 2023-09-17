from pathlib import Path

from modal import Image, Mount, Stub, asgi_app, gpu, method

import os 
### Reference: https://modal.com/docs/guide/ex/stable_diffusion_xl
### RUN:  modal run refiner.py --prompt "A mushroom kingdom"


"""
Define a container image

To take advantage of Modal’s blazing fast cold-start times, we’ll need to download our model weights inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.

Tip: avoid using global variables in this function to ensure the download step detects model changes and triggers a rebuild.
"""
def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0", ignore_patterns=ignore
    )


image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
    )
    .run_function(download_models)
)

stub = Stub("stable-diffusion-xl", image=image)

"""
Load model and run inference

The container lifecycle __enter__ function loads the model at startup. Then, we evaluate it in the run_inference function.

To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.
"""

@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    def __enter__(self):
        import torch
        from diffusers import DiffusionPipeline

        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # These suggested compile commands actually increase inference time, but may be mis-used.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    @method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        import io

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes

"""
And this is our entrypoint; where the CLI is invoked. Explore CLI options with: 
modal run stable_diffusion_xl.py --prompt 'An astronaut riding a green horse'
"""

@stub.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().inference.remote(prompt)
    print("Using prompt: ", prompt)


    dir = "./tmp/stable-diffusion-xl"
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = prompt.strip().replace(" ", '-')
    output_path =os.path.join(dir , (filename + ".png"))
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)



"""
A user interface

Here we ship a simple web application that exposes a front-end (written in Alpine.js) for our backend deployment.

The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.

We can deploy this with modal deploy stable_diffusion_xl.py."""

# Deploy as a web app with an user interface
### RUN: modal deploy refiner.py 


frontend_path = Path(__file__).parent / "frontend"


@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@asgi_app()
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.get("/infer/{prompt}")
    async def infer(prompt: str):
        from fastapi.responses import Response

        image_bytes = Model().inference.remote(prompt)

        return Response(image_bytes, media_type="image/png")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
