### 1. Generating Finetuning Labels using GPT

The process involves using GPT to analyze dashcam footage and generate detailed reasoning for collision prediction. Here's how it works:

### 1.1 Data Preparation

- The system takes sequences of dashcam images (typically 5 frames per sample) from the `train.csv` file, which includes metadata like the event time and target labels (collision or no collision).
- Each sample has a binary label (collision/no collision), determined by the presence of an incident in the dashcam footage.
- We experimented with various hyperparameters for frame extraction, including different values for the number of frames (`NUM_FRAMES`), the time interval between frames (`TIME_BETWEEN_FRAMES`), and the image resolution. These hyperparameters were adjusted to balance computational efficiency and model performance.
- Our final configuration was selected because it provided the best trade-off between performance and GPU memory usage, avoiding out-of-memory (OOM) errors during training.
- The frames are extracted using the `ImageDataset` class, which is designed as follows:

#### The `ImageDataset` Class

- **Purpose:** Efficiently loads video frames as image sequences for model training and evaluation.
- **Initialization:** Takes the following parameters:

  - `data`: A DataFrame containing video metadata and labels.
  - `videos_folder`: Path to the directory containing video files.
  - `num_frames`: Number of frames to extract from each video.
  - `time_between_frames`: Interval (in seconds) between each frame.
  - `transform`: Optional image transformations (resize, normalization).

- **Frame Extraction:** Uses the `get_multiple_frames` method:

  - If a specific event time is provided, frames are extracted relative to that time.
  - If the time is not specified, frames are extracted from the end of the video.
  - Each frame is resized and transformed using the specified `transform` function.

- This class was optimized to minimize memory usage while maintaining consistent input size and image quality.

#### 1.2 GPT Prompt Structure

- The system uses a structured prompt that asks GPT to analyze the following factors:

  - Vehicle Positions
  - Trajectories
  - Nearby Vehicles and Traffic
  - Traffic Signals
  - Road Conditions and Visibility

- The prompt dynamically changes based on whether a collision occurred in the scenario. The prompt is as follows:

```plaintext
You are an expert in accident reconstruction and traffic analysis. You will analyze a sequence of dashcam images with a chain of thought reasoning to determine whether there is an immediate threat of vehicle collision. In this particular scenario, there was {'a collision' if label else 'no collision'}. Consider each of the following factors:

1. Vehicle Positions: Identify the locations of all vehicles in each frame and how they change over time.
2. Trajectories: Determine the direction, speed, and acceleration of each vehicle by comparing their positions across frames.
3. Nearby Vehicles and Traffic: Identify surrounding vehicles, pedestrians, and any traffic congestion that could impact movement.
4. Traffic Signals: Consider whether traffic signals indicate a stop, go, or caution state and how that affects the vehicle interactions.
5. Road Conditions and Visibility: Note any obstructions, road markings, or weather conditions that could contribute to the situation.
```

#### 1.3 Label Generation Process

```python
import openai

def generate_response(conversation):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
        n=1,
    )
    return response.choices[0].message.content.strip()
```

- The response format is structured as follows:

```
<reasoning>
[Detailed analysis of the factors]
</reasoning>
<answer>
[Yes/No]
</answer>
```

#### Example GPT-Generated Reasoning (Collision Scenario)

**Prompt:** The system displayed a sequence of dashcam images indicating a potential collision. The model was tasked with analyzing the situation based on vehicle positions, trajectories, nearby traffic, signals, and road conditions.

**Generated Reasoning:**
The images show a silver SUV in front, with a white convertible moving beside it on the right. Other vehicles are visible to the left and right, indicating a busy intersection. The white convertible is progressing forward in each frame, moving into the intersection, while the silver SUV remains stationary initially. The traffic seems dense, with multiple vehicles in close proximity, and the white convertible is attempting to overtake or change lanes. The traffic signal is red in all frames, indicating that vehicles should not be moving forward into the intersection. It is daytime with no apparent weather issues affecting visibility, and road markings suggest a designated path for vehicles, but the congestion may be affecting adherence.

**Conclusion:** The white convertible is potentially violating the red light by moving forward, which could lead to a collision. The close proximity of all vehicles, with the convertible overtaking from the right at a red light, increases the collision risk. To avoid accidents, vehicles need to respect traffic signals and lane boundaries, particularly in congested areas.

**Predicted Label:** Yes (Collision)

This example illustrates how the model uses a structured chain-of-thought process to assess the situation, identifying critical factors such as vehicle trajectories, traffic signals, and potential hazards.

### 2: Fine-Tuning Models with Generated Labels

#### 2.1 Model Architecture

- The following models were fine-tuned for collision prediction:

  - LLaMA 3.2 (20250405)
  - LLaMA 3.2 Base
  - Qwen 2

#### 2.2 Training Configuration

- Training was conducted using the Unsloth optimization framework, which supports fine-tuning of vision-language models using a highly optimized pipeline.
- We utilized 4-bit quantized models to significantly reduce GPU memory usage without sacrificing model performance. This was essential to fit the models within available GPU VRAM.
- Mixed-precision training was enabled using 4-bit quantization for both model weights and activations, leveraging the `bnb-4bit` configuration in Unsloth.
- Gradient checkpointing was applied to further reduce memory usage. This allows the model to save memory by recomputing intermediate values during the backward pass.
- Model fine-tuning was performed using the following key hyperparameters:

  - Learning Rate: 2e-4 (adjusted with a linear scheduler)
  - Optimizer: AdamW (decoupled weight decay with epsilon stabilization)
  - Weight Decay: 0.01
  - Batch Size: 1 (per device, adjusted for VRAM constraints)
  - Gradient Accumulation Steps: 4 (to simulate a larger batch size)
  - Maximum Steps: 30 (for efficient experimentation)
  - Mixed-Precision: 4-bit quantization with LoRA (Low-Rank Adaptation) layers enabled

- The LoRA configuration included:

  - Rank: 16
  - Alpha: 16
  - Dropout: 0.0 (no dropout for stable training)

- We applied custom loss functions in the `compute_loss` function, integrating both standard cross-entropy loss and an accuracy penalty. The accuracy penalty ensures that the model learns to prioritize correct predictions over simply matching token-level outputs.

#### 2.3 Data Augmentation and Frame Selection

- Data augmentation was a critical aspect of our training process to improve model robustness.
- We experimented with various frame configurations:

  - Number of Frames: 5, 10, 15, and 20
  - Time Between Frames: 0.5s, 1.0s, and 1.5s

- The final configuration used was 5 frames with a 1.0s interval, balancing computational cost and model performance.
- The images were resized using a standard resolution of (480, 854) to ensure consistent input size.
- We also experimented with different image transformations applied to the frames using the PyTorch `transforms` library, including:

  - Resizing for consistent input dimensions.
  - Normalization to standardize pixel values.
  - Data augmentation strategies were explored (e.g., random crop, color jitter) but were avoided in the final setup to maintain input consistency for collision prediction.

- Frame selection was handled by the `ImageDataset` class, which efficiently loads and preprocesses frames based on the specified configuration.
- Our approach ensured that the model learned to recognize critical collision indicators across different frame sequences and time intervals, improving its generalization.

#### 2.4 Validation and Evaluation

- The models were evaluated using binary classification metrics (accuracy, precision, recall, F1-score), with the following results:

| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| -------------------- | -------- | --------- | ------ | -------- | ------- |
| LLaMA 3.2 (20250405) | 50.00%   | 0.5       | 100.0% | 0.667    | 0.5     |
| LLaMA 3.2 Base       | 54.67%   | 1.0       | 9.33%  | 0.171    | 0.5     |
| Qwen 2               | 50.00%   | 0.5       | 100.0% | 0.667    | 0.5     |

- The evaluation is automated using the `extract_label` and `compute_metrics` functions, which ensure consistency in calculating performance metrics.

#### 2.5 Performance Optimization

- The models leverage gradient checkpointing for memory efficiency, allowing larger models to fit into available GPU memory.
- Training utilizes mixed-precision (4-bit quantization) for faster computations and reduced memory footprint.
- The learning rate is optimized using the AdamW optimizer with weight decay, ensuring stable gradient updates during training.

### 3: Reward Optimization Using GRPO

#### 3.1 Reward Functions

- The model training uses multiple reward functions to ensure high-quality outputs:

  - **Correctness Reward:** Compares the model's answer with the ground truth and assigns a score of 2.0 for correct answers.
  - **Strict Format Reward:** Verifies that the response follows the structured format (`<reasoning>...</reasoning><answer>...</answer>`) and rewards compliance.
  - **Soft Format Reward:** A more flexible format check that only requires the presence of reasoning and answer tags.
  - **XML Count Reward:** Evaluates the XML tag structure for completeness, rewarding well-formed outputs.
  - **Integer Reward:** Checks if the response contains a valid numerical value, useful for ensuring quantitative answers.

#### 3.2 GRPO Training Configuration

- The model is trained using GRPO (Generalized Reward Policy Optimization), a reinforcement learning approach optimized for text generation tasks.
- Configuration includes:

  - Learning rate: 5e-6
  - Optimizer: Paged AdamW 8-bit for memory efficiency
  - Weight decay: 0.1
  - Max gradient norm: 0.1 for stable gradient updates
  - Training steps: 250

#### 3.3 Fine-Tuning with GRPO

- Training involves sampling multiple responses for each prompt and evaluating them using the reward functions.
- The model adjusts its parameters to maximize the average reward score across the sampled responses.
- The approach encourages the model to generate responses that are not only accurate but also well-structured and compliant with XML formatting.
