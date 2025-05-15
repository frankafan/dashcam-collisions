#!/usr/bin/env python
# coding: utf-8

# #### Dependencies

# In[35]:


# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab and Kaggle notebooks! Otherwise use pip install unsloth
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
#     !pip install --no-deps cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
#     !pip install --no-deps unsloth


# In[36]:


import unsloth
from unsloth import FastVisionModel  # FastLanguageModel for LLMs


# In[ ]:


import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from transformers import TextStreamer
import re
import torch.nn.functional as F
from itertools import product


# #### Hyperparameters

# In[ ]:


DATA_DIR = "data"
OUTPUT_DIR = (
    f"evaluation/llama_3.2_11B_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
MODEL_NAME = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"


# In[ ]:


# Hyperparameters
NUM_FRAMES = [10, 15, 20]  # Number of frames to extract from the video
TIME_BETWEEN_FRAMES = [0.5, 1.0, 1.5]  # Time between frames in seconds


# #### Load Data

# In[ ]:


train_csv_path = f"{DATA_DIR}/train.csv"
train_videos_folder = f"{DATA_DIR}/train/"

df = pd.read_csv(train_csv_path)
df.head()


# In[ ]:


class ImageDataset(Dataset):
    def __init__(
        self, data, videos_folder, num_frames, time_between_frames, transform=None
    ):
        self.data = data
        self.videos_folder = videos_folder
        self.transform = transform  # Any image transformations (e.g., augmentations)
        self.num_frames = num_frames
        self.time_between_frames = time_between_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        vid_path = os.path.join(
            self.videos_folder, f"{str(int(row['id'])).zfill(5)}.mp4"
        )

        time = None if np.isnan(row["time_of_event"]) else row["time_of_event"]
        images = self.get_multiple_frames(vid_path, time)

        label = row["target"]

        # Apply transformations
        if self.transform:
            images = [self.transform(image) for image in images]

        return images, torch.tensor(label), row["id"]  # Convert label to tensor

    def get_frame(self, video_path, time):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)

        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError(f"Failed to read frame at {time} seconds.")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def get_multiple_frames(self, video_path, time):
        time_between_frames = self.time_between_frames

        if time == 0:
            # Get the first frame
            frame = self.get_frame(video_path, time)
            return [frame]

        if time is None:
            # Get the last frames
            cap = cv2.VideoCapture(video_path)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
            fps = cap.get(cv2.CAP_PROP_FPS)
            time = frame_count / fps
            cap.release()

        frames = []
        for i in range(self.num_frames):
            try:
                frame = self.get_frame(video_path, time - i * time_between_frames)
                frames.append(frame)
            except ValueError:
                break
        return frames


transform = transforms.Compose(
    [
        transforms.Resize((480 // 3, 854 // 3)),  # Resize to a standard size
    ]
)


# #### Load Models

# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",  # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",  # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
    "unsloth/Pixtral-12B-2409-bnb-4bit",  # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",  # Pixtral base model
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",  # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",  # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    # "unsloth/Llama-3.2-11B-Vision-Instruct",
    MODEL_NAME,
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)


# In[ ]:


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# #### Convert Datasets

# In[ ]:


SYSTEM_PROMPT = """
You are an expert in accident reconstruction and traffic analysis. You will analyze a sequence of dashcam images with a chain of thought reasoning to determine whether there is an immediate threat of vehicle collision. Consider each of the following factors:

1. Vehicle Positions: Identify the locations of all vehicles in each frame and how they change over time.
2. Trajectories: Determine the direction, speed, and acceleration of each vehicle by comparing their positions across frames.
3. Nearby Vehicles and Traffic: Identify surrounding vehicles, pedestrians, and any traffic congestion that could impact movement.
4. Traffic Signals: Consider whether traffic signals indicate a stop, go, or caution state and how that affects the vehicle interactions. Pay special attention on whether vehicles are vialating or obeying traffic signal rules.
5. Road Conditions and Visibility: Note any obstructions, road markings, or weather conditions that could contribute to the situation.
"""

SYSTEM_FORMAT_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def convert_to_conversation(images):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "text", "text": SYSTEM_FORMAT_PROMPT},
                *[
                    {
                        "type": "image",
                        "image": image,
                    }
                    for image in images
                ],
            ],
        },
    ]

    return {"messages": conversation}


# #### Generate Validation Predictions

# In[ ]:


FastVisionModel.for_inference(model)  # Enable for inference!


# In[ ]:


def extract_label(decoded_text):
    match = re.search(r"(?i)<answer>\s*(Yes|No)\s*</answer>", decoded_text)
    if match:
        answer = match.group(1).strip().lower()
        if answer == "yes":
            return 1
        elif answer == "no":
            return 0
    return None


# In[ ]:


os.makedirs(OUTPUT_DIR, exist_ok=True)

for num_frames, time_between_frames in tqdm(product(NUM_FRAMES, TIME_BETWEEN_FRAMES)):
    print(
        f"Evaluating with num_frames={num_frames} and time_between_frames={time_between_frames}"
    )

    results = []

    dataset = ImageDataset(
        df,
        train_videos_folder,
        num_frames=num_frames,
        time_between_frames=time_between_frames,
        transform=transform,
    )

    for image, target, id in tqdm(dataset):
        conversation = convert_to_conversation(image)
        input_text = tokenizer.apply_chat_template(
            conversation["messages"], add_generation_prompt=True
        )

        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        output_tokens = model.generate(
            **inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1
        )
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        extracted_label = extract_label(output_text)

        # Store more detailed results
        results.append(
            {
                "id": id,
                "model": MODEL_NAME,
                "num_frames": num_frames,
                "time_between_frames": time_between_frames,
                "target": target,
                "response": output_text,
                "extracted_label": extracted_label,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    df = pd.DataFrame(results)
    output_filename = f"{OUTPUT_DIR}/results.csv"
    df.to_csv(output_filename, mode="a", index=False)
    print(f"Results saved to: {output_filename}")

