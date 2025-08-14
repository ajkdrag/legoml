# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LegoML is a modular PyTorch machine learning framework designed around a configuration-driven architecture. The project uses a "Lego blocks" approach where neural network components, optimizers, schedulers, and datasets are defined as configurable building blocks that can be easily composed.

## Key Commands

### Running Experiments
```bash
# Run the main training experiment
python main.py

# Run testing on trained models
python -c "from legoml.experiments import test_tiny_cnn_baseline; test_tiny_cnn_baseline()"

# Test specific checkpoint
python -c "from legoml.experiments import test_tiny_cnn_baseline; test_tiny_cnn_baseline('models/tiny_cnn_baseline_epoch_1.pt')"

# The project uses uv for dependency management
uv sync  # Install dependencies
uv run python main.py  # Run with uv
```

### Development Setup
```bash
# Install dependencies (including dev dependencies)
uv sync --group dev

# The project uses structured logging - check logs for experiment progress
```

## Architecture Overview

### Configuration System
The project centers around a configuration-driven architecture:

- **BaseConfig** (`src/legoml/configs/base.py`): Abstract base classes for all configurations
- **Composable Configs**: Model, Dataset, Optimizer, Scheduler configs that implement `build()` methods
- **JSON Serializable**: All configs can be saved/loaded as JSON for experiment tracking

### Core Components

1. **Trainers** (`src/legoml/configs/trainers/`):
   - `SupervisedTrainer`: Handles training loops, validation, checkpointing
   - Built from `SupervisedTrainerConfig` which composes model, dataset, optimizer configs

2. **Models** (`src/legoml/configs/models/`):
   - Modular CNN architectures built from conv blocks
   - Example: `TinyCNN` with configurable conv blocks and FC layers

3. **Datasets** (`src/legoml/configs/datasets/`):
   - Currently supports MNIST
   - Handles data loading, augmentation, train/val/test splits

4. **Building Blocks** (`src/legoml/configs/blocks/`):
   - Reusable components: convolution blocks, activations, pooling
   - Composable via configuration

### Testing and Evaluation

The framework provides comprehensive testing capabilities:

1. **SupervisedTrainer.test()**: Test method for existing trainers
2. **TestConfig/TestTrainer** (`src/legoml/configs/trainers/test.py`):
   - Dedicated config for test-only evaluation
   - Loads models from checkpoints
   - Supports detailed metrics including per-class metrics and confusion matrices
3. **Checkpoint Utilities** (`src/legoml/utils/checkpoints.py`):
   - `load_checkpoint()`: Load checkpoint files
   - `load_model_from_checkpoint()`: Load model weights from checkpoint
   - `checkpoint_info()`: Display checkpoint metadata
   - `find_latest_checkpoint()`: Find most recent checkpoint in directory

### Experiment Pattern
Experiments are defined as functions that:
1. **Training**: Create a `SupervisedTrainerConfig`, call `config.build()`, run `trainer.train()`
2. **Testing**: Create a `TestConfig`, call `config.build()`, run `tester.test()`

Example experiments in `experiments.py`:
- `train_tiny_cnn_baseline()`: Training experiment
- `test_tiny_cnn_baseline()`: Testing experiment (auto-finds latest checkpoint)
- `test_from_checkpoint()`: Generic testing function

### Logging and Monitoring
- Uses `structlog` for structured logging
- Automatic experiment tracking with metrics
- Model checkpointing every N epochs
- Progress bars via `tqdm`

### Device Support
- Automatically detects available devices (CUDA/MPS/CPU)
- Default uses MPS on macOS for Apple Silicon acceleration

## File Structure
- `main.py`: Entry point that runs the baseline experiment
- `src/legoml/`: Main package
  - `configs/`: All configuration classes organized by component type
  - `experiments.py`: Experiment definitions
  - `utils/`: Logging and utility functions
- `data/`: Dataset storage (MNIST included)
- `models/`: Saved model checkpoints

## Development Notes

- No formal test suite currently exists
- Uses lazy linear layers (`nn.LazyLinear`) to automatically infer dimensions
- Configuration validation happens at build time
- All configs inherit from `BaseConfig` and implement abstract `build()` methods