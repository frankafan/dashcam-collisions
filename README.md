# Nexar Collision Prediction Challenge

This project implements a computer vision-based collision prediction system using the LLaMA 3.2 Vision model. The system analyzes video frames to predict potential collisions, leveraging state-of-the-art vision-language models and optimization techniques.

## Project Overview

The project uses the Unsloth-optimized LLaMA 3.2 Vision model to analyze video frames and predict collision probabilities. It includes both training and evaluation pipelines, with support for various optimization techniques and evaluation metrics.

## Project Structure

- `data/` - Directory containing training data and videos
- `outputs/` - Directory for model outputs and checkpoints
- `evaluation/` - Evaluation scripts and results
- `examples/` - Example usage and demonstrations

### Key Files

- `unsloth_finetuning.py` - Main training script with Unsloth optimization
- `unsloth_evaluate.py` - Evaluation script for the finetuned model
- `openai_evaluate.py` - Evaluation script for OpenAI models
- `data.ipynb` - Data preparation and preprocessing notebook
- `prediction_result_analysis.ipynb` - Analysis of model predictions

## Usage

### Training

To train the model using Unsloth optimization:

```bash
python unsloth_finetuning.py
```

The script will:
- Load and preprocess the video data
- Initialize the LLaMA 3.2 Vision model
- Apply Unsloth optimizations
- Train the model on the provided dataset
- Save checkpoints in the outputs directory

### Evaluation

To evaluate the model:

```bash
python unsloth_evaluate.py
```

For OpenAI model evaluation:

```bash
python openai_evaluate.py
```

## Model Details

- Base Model: LLaMA 3.2 11B Vision Instruct
- Optimization: Unsloth (4-bit quantization)
- Input: Video frames (5 frames per sample)
- Output: Collision prediction probability

## Data Format

The training data consists of:
- Video files in MP4 format
- CSV file with annotations including:
  - Video ID
  - Time of event
  - Target (collision/no collision)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Unsloth team for optimization tools
- Meta AI for the LLaMA 3.2 Vision model
- Nexar for the challenge and dataset
