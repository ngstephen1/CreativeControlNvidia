# controlnet_smoke_test.py
import torch
from diffusers.utils import load_image
from controlnet_bria import BriaControlNetModel
from pipeline_bria_controlnet import BriaControlNetPipeline
import PIL.Image as Image

# ---- device selection (Apple M1/M2 GPU if available, else CPU) ----
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# ---- helper for BRIA 1024 ratios (copied from model card) ----
RATIO_CONFIGS_1024 = {
    0.6666666666666666: {"width": 832, "height": 1248},
    0.7432432432432432: {"width": 880, "height": 1184},
    0.8028169014084507: {"width": 912, "height": 1136},
    1.0: {"width": 1024, "height": 1024},
    1.2456140350877194: {"width": 1136, "height": 912},
    1.3454545454545455: {"width": 1184, "height": 880},
    1.4339622641509433: {"width": 1216, "height": 848},
    1.5: {"width": 1248, "height": 832},
    1.5490196078431373: {"width": 1264, "height": 816},
    1.62: {"width": 1296, "height": 800},
    1.7708333333333333: {"width": 1360, "height": 768},
}


def resize_img(control_image: Image.Image) -> Image.Image:
    image_ratio = control_image.width / control_image.height
    ratio = min(RATIO_CONFIGS_1024.keys(), key=lambda k: abs(k - image_ratio))
    to_height = RATIO_CONFIGS_1024[ratio]["height"]
    to_width = RATIO_CONFIGS_1024[ratio]["width"]
    resized_image = control_image.resize(
        (to_width, to_height), resample=Image.Resampling.LANCZOS
    )
    return resized_image


def main():
    base_model = "briaai/BRIA-3.2"
    controlnet_model = "briaai/BRIA-3.2-ControlNet-Union"

    print("Loading ControlNet…")
    controlnet = BriaControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    print("Loading BRIA-3.2 pipeline…")
    pipe = BriaControlNetPipeline.from_pretrained(
        base_model,
        revision="pre_diffusers_support",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )

    pipe = pipe.to(device=device)

    # Use their demo canny image
    print("Downloading control image (canny)…")
    control_image_canny = load_image(
        "https://huggingface.co/briaai/BRIA-3.2-ControlNet-Union/resolve/main/images/canny.jpg"
    )
    control_image_canny = resize_img(control_image_canny)
    width, height = control_image_canny.size

    prompt = (
        "In a serene living room, someone rests on a sapphire blue couch, "
        "diligently drawing in a rose-tinted notebook."
    )
    controlnet_conditioning_scale = 1.0
    control_mode = 1  # 1 = canny

    generator = torch.Generator(device=device).manual_seed(555)

    print("Running inference (this may take a while on CPU)…")
    image = pipe(
        prompt,
        control_image=control_image_canny,
        control_mode=control_mode,
        width=width,
        height=height,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=12,  # lower than 50 to keep it lighter
        max_sequence_length=128,
        guidance_scale=5.0,
        generator=generator,
        negative_prompt=(
            "Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,"
            "Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate"
        ),
    ).images[0]

    out_path = "controlnet_union_test.png"
    image.save(out_path)
    print(f"✅ Done. Saved image to {out_path}")


if __name__ == "__main__":
    main()