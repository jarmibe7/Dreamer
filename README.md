# Dreamer v1 Implementation
## Jared Berry

A PyTorch implementation of the Dreamer v1 algorithm for model-based reinforcement learning in visual environments using Gymnasium. This project was associated with ME_395: ML for Mechanical Sciences at Northwestern University.

## Overview

This project implements Dreamer v1, which learns to solve RL tasks by:
1. **Learning a world model** (RSSM - Recurrent State Space Model) that predicts future observations and rewards from actions
2. **Planning in imagination** by rolling out the learned world model without environment interaction
3. **Training an actor-critic policy** using imagined trajectories to maximize long-horizon returns

The implementation supports both image-based and feature-augmented observations, with configurable architectures and hyperparameters via YAML config files.

## Project Structure

```
├── dreamer/                    # Main package
│   ├── __init__.py
│   ├── dreamer.py              # Dreamer trainer (main Dreamer algorithm)
│   ├── trainer.py              # Base RSSM trainer (independent world model learning)
│   ├── model_utils.py          # Replay buffer, evaluation, plotting utilities
│   ├── runtime_utils.py        # Environment setup, model building, config handling
│   ├── model/
│   │   ├── rssm.py             # RSSM architecture (encoder/decoder + dynamics)
│   │   ├── encoder.py          # Convolutional image encoder
│   │   ├── decoder.py          # Convolutional image decoder
│   │   ├── loss.py             # RSSM loss (reconstruction + KLD)
│   │   └── __init__.py
│   └── __init__.py
├── config/                     # Configuration files
│   ├── train_rssm.yaml         # Config for world model pre-training
│   └── train_dreamer.yaml      # Config for full Dreamer training
├── train_rssm.py               # Entry point: train world model only
├── train_dreamer.py            # Entry point: train full Dreamer agent
├── runs/                       # Training outputs (checkpoints, plots, videos)
└── README.md
```

## Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install swig
   pip install -r requirements.txt
   ```

## Usage

### Train World Model Only (RSSM)

Pre-train the world model to accurately predict observations and rewards:

```bash
python train_rssm.py --config config/train_rssm.yaml
```

### Train Full Dreamer Agent

Train the complete Dreamer agent (world model + actor + value network):

```bash
python train_dreamer.py --config config/train_dreamer.yaml
```

### Configuration

Modify `config/train_dreamer.yaml` to change:
- **Environment**: environment name and settings
- **Model architecture**: RSSM settings, encoder/decoder sizes, latent dimensions, prediction horizon
- **Training**: learning rates, batch size, number of epochs, buffer capacity
- **Loss weights**: reconstruction multiplier, KLD weight, reward loss weight
- **Dreamer hyperparameters**: imagination horizon, discount factor, actor/value learning rates

### Outputs

Training outputs are saved to `runs/{env_name}/{timestamp}/`:
- `config.yaml` - Training configuration used
- `checkpoints/` - Model checkpoints saved periodically
- `plots/` - Loss curves and episode return plots
- `videos/` - Evaluation episode videos (ground truth vs. predicted observations)

## Key Implementation Details

### RSSM (Recurrent State Space Model)
- **Encoder**: ConvNet that compresses observations into latent embeddings
- **Prior**: RNN that predicts next latent state from current state + action
- **Posterior**: Refines latent state using actual next observation
- **Decoder**: Reconstructs observations from latent states
- **Reward head**: Predicts rewards from latent states

### Dreamer (Actor-Critic)
- **Actor**: Learns to propose actions that maximize imagined returns
- **Value network**: Estimates long-horizon return from latent states
- **Imagination loop**: Rolls RSSM forward using actor's actions (no environment needed)
- **Lambda returns**: Combines N-step returns with value bootstrapping

### Training Algorithm
1. **Collect episodes** using current policy in environment
2. **World model update**: Minimize reconstruction and KLD losses
3. **Imagination**: Roll RSSM forward from posterior latent states
4. **Actor update**: Maximize expected return + entropy in imagined trajectories
5. **Value update**: Minimize MSE loss on imagined returns

## Requirements

Python 3.8+

See `requirements.txt` for package dependencies.

## Citations

LLM-based tools were used to assist in the production of this code.

```bibtex
@misc{hafner2020dreamcontrollearningbehaviors,
      title={Dream to Control: Learning Behaviors by Latent Imagination}, 
      author={Danijar Hafner and Timothy Lillicrap and Jimmy Ba and Mohammad Norouzi},
      year={2020},
      eprint={1912.01603},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1912.01603}, 
}
```

