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

### Node-Based Configuration System
The project centers around a node-based architecture using dataclasses:

- **Node** (`src/legoml/interfaces/nodes.py`): Abstract base class for all configurable components with `build()` methods
- **Composable Nodes**: Model, Dataset, Optimizer, Scheduler nodes that can be easily composed and forked
- **JSON Serializable**: All nodes can be saved/loaded via `to_dict()` for experiment tracking

### Core Node Types

1. **BlockNode**: Base for models, layers and blocks
   - `ModelNode`: Neural network models (e.g., `TinyCNNNode`)
   - `ActivationNode`: Activation functions 
   - `PoolingNode`: Pooling layers

2. **Training Components**:
   - `OptimizerNode`: Optimizers with parameter binding
   - `SchedulerNode`: Learning rate schedulers
   - `MetricNode`: Training and evaluation metrics

3. **Data Components**:
   - `DatasetNode`: Dataset definitions 
   - `DataLoaderNode`: DataLoader configurations
   - `CollatorNode`: Custom data collation

### Core Implementation Areas

1. **Tasks** (`src/legoml/tasks/`):
   - `TrainerForImageClassificationNode`: Complete training pipeline with metrics, checkpointing
   - `EvaluatorForImageClassificationNode`: Evaluation-only pipeline for testing

2. **Models** (`src/legoml/models/`):
   - Modular CNN architectures built from composable blocks
   - Example: `TinyCNNNode` with configurable conv blocks and FC layers

3. **Datasets** (`src/legoml/datasets/`):
   - Currently supports MNIST classification
   - Handles data loading, augmentation, train/val/test splits

4. **Building Blocks** (`src/legoml/blocks/`):
   - Reusable components: convolution blocks, activations, pooling
   - Composable via node configuration
   - Encoders for more complex architectural patterns

5. **Metrics** (`src/legoml/metrics/`):
   - Standardized metric interface (`Metric` ABC)
   - Classification metrics (accuracy, loss tracking)
   - Extensible for custom metrics

### Testing and Evaluation

The framework provides comprehensive testing capabilities:

1. **TrainerForImageClassification.train()**: Complete training pipeline with built-in evaluation
2. **EvaluatorForImageClassificationNode** (`src/legoml/tasks/eval_img_clf.py`):
   - Dedicated node for test-only evaluation
   - Loads models from checkpoints
   - Supports detailed metrics including per-class metrics and confusion matrices
3. **Checkpoint Utilities** (`src/legoml/utils/checkpoints.py`):
   - `load_checkpoint()`: Load checkpoint files
   - `load_model_from_checkpoint()`: Load model weights from checkpoint
   - `checkpoint_info()`: Display checkpoint metadata
   - `find_latest_checkpoint()`: Find most recent checkpoint in directory
4. **Builder Utilities** (`src/legoml/utils/builders.py`):
   - `build_model()`: Construct models from nodes with checkpoint loading
   - `build_dataloader()`: Create dataloaders from dataset and dataloader nodes
   - `build_optimizer_and_scheduler()`: Initialize optimizers and schedulers
   - `build_metrics()`: Initialize metric objects from metric nodes

### Experiment Pattern
Experiments are defined as functions that:
1. **Training**: Create a `TrainerForImageClassificationNode`, call `node.build()`, run `trainer.train()`
2. **Testing**: Create an `EvaluatorForImageClassificationNode`, call `node.build()`, run `evaluator.eval()`

Example experiments in `experiments.py`:
- `train_tiny_cnn_baseline()`: Training experiment using nodes
- Node composition allows easy experimentation with different architectures and hyperparameters

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
  - `interfaces/`: Base node classes and metric interfaces
    - `nodes.py`: Node base classes (Node, BlockNode, ModelNode, etc.)
    - `metrics.py`: Metric abstract base class
  - `tasks/`: Training and evaluation pipelines
    - `train_img_clf.py`: Image classification training node
    - `eval_img_clf.py`: Image classification evaluation node
  - `models/`: Neural network model implementations
    - `tinycnn.py`: TinyCNN model node and implementation
  - `blocks/`: Reusable building blocks
    - `convs.py`: Convolutional block nodes
    - `activations.py`: Activation function nodes
    - `pooling.py`: Pooling layer nodes
    - `encoders/`: More complex encoder architectures
  - `datasets/`: Dataset implementations
    - `mnist_clf.py`: MNIST classification dataset node
  - `metrics/`: Metric implementations
    - `accuracy.py`: Accuracy metrics
    - `classification.py`: Classification-specific metrics
    - `loss.py`: Loss tracking metrics
  - `dataloaders.py`: DataLoader node implementations
  - `optimizers.py`: Optimizer node implementations
  - `schedulers.py`: Learning rate scheduler node implementations
  - `experiments.py`: Experiment definitions using nodes
  - `utils/`: Utility functions
    - `builders.py`: Builder functions for nodes
    - `checkpoints.py`: Checkpoint utilities
    - `logging.py`: Structured logging setup
    - `misc.py`: Miscellaneous utilities
- `data/`: Dataset storage (MNIST included)
- `models/`: Saved model checkpoints

## Development Notes

- No formal test suite currently exists
- Uses lazy linear layers (`nn.LazyLinear`) to automatically infer dimensions
- Node validation happens at build time via abstract `build()` methods
- All nodes inherit from `Node` base class and are dataclass-based
- Nodes support forking (`fork()`) to create variations with modified parameters
- Node composition enables modular experiment design
- Interface-based design with `Metric` ABC for extensible metrics