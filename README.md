<h2 align="center">Low Rank Adaptation Methods: Efficiently Fine-Tuning Large Model</h3> 

---
The world of machine learning/deep learning (ML/DL) is constantly evolving, and it's important to keep up with the latest advancements. One of the biggest challenges in ML/DL is model fine-tuning. Fine-tuning a model requires a lot of computational power and time. Additionally, the generated models often require continuous updates, which can be time-consuming and inefficient. This is where LoRA comes in - a new technology that can help solve these issues. 


<h3>Installation</h4>

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependency.

```bash
pip install -r requirements.txt
```
Note: Make sure you have created a virtual environment. 

### Usage

#### Fine-tuning your model
```bash
export MODEL_NAME = {name_your_model}
export OUTPUT_DIR = "./lora"
export DATASET_NAME = {path/to/your/dataset}
```

Then

```bash
accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5' \
  --dataset_name='lambdalabs/pokemon-blip-captions' \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir='/content/lora' \
  --checkpointing_steps=500 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337
```

#### Inference 

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from src.utils.utils import image_grid


model_base = "lambdalabs/sd-pokemon-diffusers"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs("megrxu/pokemon-lora")
pipe.to("cuda")


images = pipe(
    "A painting of Bulbasaur in a lush, green forest, surrounded by tall trees and blooming flowers. The little Pokémon is sitting on a mossy rock, with its bulb glowing brightly in the sunlight filtering through the leaves. Its eyes are closed, as if it's basking in the warmth and energy of the sun. The painting is done in a detailed, realistic style, with each leaf and blade of grass carefully rendered.", 
    num_inference_steps=30, 
    guidance_scale=7.5, 
    num_images_per_prompt=3,
    negative_prompt="Ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy.",
    cross_attention_kwatgs={"scale": 0.5}
)

image_grid(images[0], 1, 3)
```