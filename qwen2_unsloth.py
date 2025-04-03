#!/usr/bin/env python
# coding: utf-8

# #### Dependencies

# In[1]:


# get_ipython().run_cell_magic('capture', '', 'import os\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth\nelse:\n    # Do this only in Colab and Kaggle notebooks! Otherwise use pip install unsloth\n    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton\n    !pip install --no-deps cut_cross_entropy unsloth_zoo\n    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer\n    !pip install --no-deps unsloth\n')


# In[ ]:


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
from transformers import TextStreamer, AutoImageProcessor
import re


# #### Hyperparameters

# In[4]:


DATA_DIR = "data"
OUTPUT_DIR = "outputs/qwen_2"


# In[5]:


# Hyperparameters
NUM_FRAMES = 5  # Number of frames to extract from the video
TIME_BETWEEN_FRAMES = 1  # Time between frames in seconds


# #### Load Data

# In[ ]:


train_csv_path = f"{DATA_DIR}/train.csv"
train_videos_folder = f"{DATA_DIR}/train/"

df = pd.read_csv(train_csv_path)
df.head()


# In[7]:


class ImageDataset(Dataset):
    def __init__(self, data, videos_folder, transform=None):
        self.data = data
        self.videos_folder = videos_folder
        self.transform = transform  # Any image transformations (e.g., augmentations)
        self.num_frames = NUM_FRAMES
        self.time_between_frames = TIME_BETWEEN_FRAMES

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
        transforms.Resize((480, 854)),  # Resize to a standard size
    ]
)


# In[8]:


train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["target"]
)

train_dataset = ImageDataset(train_df, train_videos_folder, transform=transform)
val_dataset = ImageDataset(val_df, train_videos_folder, transform=transform)


# In[ ]:


images, label, id = train_dataset[np.random.randint(0, len(train_dataset))]

num_images = len(images)
fig, axes = plt.subplots(1, num_images, figsize=(24, 3))

if num_images == 1:
    axes.imshow(images[0])
    axes.axis("off")
else:
    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis("off")

plt.suptitle(f"Label: {int(label)}, ID: {int(id)}")
plt.show()


# In[ ]:


reasoning_df = pd.read_csv("gpt4o_cot.csv", encoding="cp1252")
reasoning_dict = dict(zip(reasoning_df["id"], reasoning_df["response"]))

reasoning_df.head()


# #### Load Models

# In[ ]:


model_name = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"


# In[ ]:


processor = AutoImageProcessor.from_pretrained(model_name)
processor.size = {"shortest_edge": 480, "longest_edge": 854}


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
    model_name,
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

# In[13]:


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


def convert_to_conversation(images, label, id):
    reasoning = reasoning_dict.get(str(int(id)))
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
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{'Yes' if label else 'No'}\n</answer>",
                }
            ],
        },
    ]

    return {"messages": conversation}


# In[ ]:


train_converted_dataset = [
    convert_to_conversation(images, label, id)
    for images, label, id in tqdm(train_dataset)
]
val_converted_dataset = [
    convert_to_conversation(images, label, id)
    for images, label, id in tqdm(val_dataset)
]


# In[ ]:


train_converted_dataset[0]


# #### Training

# In[18]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Extract Yes/No from <answer> tags
    predicted_labels = []
    for pred in decoded_preds:
        match = re.search(r"<answer>\s*(Yes|No)\s*</answer>", pred)
        if match:
            predicted_labels.append(1 if match.group(1).strip() == "Yes" else 0)
        else:
            # Default to 0 if no clear answer found
            predicted_labels.append(0)

    # Convert to numpy arrays for metric calculation
    predicted_labels = np.array(predicted_labels)
    true_labels = labels

    accuracy = (predicted_labels == true_labels).mean()
    return {
        "accuracy": accuracy,
    }


# In[39]:


torch.cuda.empty_cache()


# In[ ]:


from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)  # Enable for training!

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
    train_dataset=train_converted_dataset,
    eval_dataset=val_converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        evaluation_strategy="steps",  # Enable evaluation
        eval_steps=5,  # Evaluate every 5 steps
        save_strategy="steps",
        save_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        max_seq_length=2048,
    ),
)


# In[ ]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[ ]:


trainer_stats = None
try:
    trainer_stats = trainer.train(resume_from_checkpoint=False)
except RuntimeError as e:
    print("Runtime error:", e)


# In[ ]:


eval_results = trainer.evaluate()
print(f"\nFinal validation loss: {eval_results['eval_loss']:.4f}")


# In[ ]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# In[ ]:


log_history = trainer.state.log_history

train_steps = [log["step"] for log in log_history if "loss" in log]
train_loss = [log["loss"] for log in log_history if "loss" in log]
eval_steps = [log["step"] for log in log_history if "eval_loss" in log]
eval_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]

plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/training_loss.png")
plt.show()


# #### Save Model

# In[ ]:


model.save_pretrained(f"{OUTPUT_DIR}/pretrained/finetuned_qwen_2/")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/pretrained/finetuned_qwen_2/")


# #### Generate Validation Predictions

# In[ ]:


FastVisionModel.for_inference(model)  # Enable for inference!


# In[ ]:


results = []

for image, target, id in tqdm(val_dataset):
    conversation = convert_to_conversation(image, label, id)
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

    # Store more detailed results
    results.append(
        {
            "id": id,
            "target": target,
            "response": output_text,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

df = pd.DataFrame(results)
output_filename = (
    f"{OUTPUT_DIR}/val_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
df.to_csv(output_filename, index=False)
print(f"Results saved to: {output_filename}")


# #### Generate Test Predictions

# In[51]:


test_csv_path = f"{DATA_DIR}/test.csv"
test_videos_folder = f"{DATA_DIR}/test/"
test_set = ImageDataset(df, train_videos_folder, transform=transform)


# In[ ]:


results = []

for image, target, id in tqdm(test_set):
    conversation = convert_to_conversation(image, label, id)
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

    # Store more detailed results
    results.append(
        {
            "id": id,
            "target": target,
            "response": output_text,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

df = pd.DataFrame(results)
output_filename = (
    f"{OUTPUT_DIR}/test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
df.to_csv(output_filename, index=False)
print(f"Results saved to: {output_filename}")
