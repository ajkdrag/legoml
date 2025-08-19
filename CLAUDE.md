# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Dependencies and Environment
- Uses `uv` for dependency management
- Install dependencies: `uv sync`
- Add dev dependencies: `uv add --dev <package>`
- Python dependencies include PyTorch, pytorch-ignite, structlog, torchvision

### Testing
- Run tests: `python -m unittest tests/test_tracking.py`
- Or run all tests: `python -m unittest discover tests`

### Running Experiments
- Main entry point: `python main.py`
- Individual experiments located in `src/legoml/exp_*.py` files

### Type Checking
- Uses mypy for type checking: `mypy src/`

## Architecture Overview

LegoML is a PyTorch-based machine learning framework built around a composable, event-driven architecture:

### Core Components

**Engine (`src/legoml/core/engine.py`)**
- Central training loop orchestrator
- Executes user-defined step functions (train_step, eval_step)
- Fires events at key lifecycle points (ENGINE_START, EPOCH_START, STEP_START, etc.)
- Manages callbacks and state

**Context (`src/legoml/core/context.py`)**  
- Holds shared training state: model, optimizer, loss_fn, device, scheduler, scaler
- Passed to step functions and callbacks for access to training components

**Callbacks (`src/legoml/core/callback.py`)**
- Protocol-based callback system for extending training behavior
- Key callbacks: EvalOnEpochEndCallback, MetricsCallback
- Respond to engine events (on_epoch_start, on_step_end, etc.)

### Neural Network Components

**Composable Nodes (`src/legoml/nn/`)**
- Node-based architecture for building models
- Base classes in `base.py`, specific implementations in subdirectories
- Examples: MLPNode, TinyCNNNode, ReluNode, NoopNode
- Nodes have `.build()` method to construct PyTorch modules

### Data Pipeline

**Data Loading (`src/legoml/data/`)**
- MNIST dataset utilities in `mnist.py`
- Batch processing utilities in `batches.py`

### Metrics and Tracking

**Metrics (`src/legoml/metrics/`)**
- Multiclass accuracy, binary classification, F1 scores
- Update/compute pattern for accumulating metrics across batches

**Experiment Tracking (`src/legoml/utils/track.py`)**
- Context manager `run()` for experiment tracking
- Logs scalars, text, and saves model artifacts
- Creates structured directory with logs/scalars.csv and artifacts

### Key Patterns

1. **Structured Step Outputs**: Step functions return `StepOutput` protocol objects (e.g., `SupervisedStepOutput`) instead of dicts for type safety
2. **Event-Driven**: Callbacks respond to engine lifecycle events
3. **Context Passing**: Shared state passed through Context dataclass
4. **Composable Models**: Build models from reusable Node components
5. **Structured Logging**: Use structlog with `get_logger(__name__)` pattern
6. **Backward Compatibility**: Engine automatically converts dict returns to `DictStepOutput` wrappers

### StepOutput System

The framework uses a `StepOutput` protocol for structured communication between step functions and callbacks:

- `SupervisedStepOutput`: For supervised learning with loss, predictions, targets, metadata
- `UnsupervisedStepOutput`: For unsupervised learning with reconstructions, embeddings
- `DictStepOutput`: Backward compatibility wrapper for dict-based returns

### Example Training Flow
```python
from legoml.core.step_output import SupervisedStepOutput

def train_step(engine, batch, context) -> SupervisedStepOutput:
    # Training logic using context.model, context.optimizer, etc.
    return SupervisedStepOutput(
        loss=loss,
        predictions=outputs.detach().cpu(),
        targets=targets.detach().cpu(),
        metadata={"loss_scalar": loss.item()}
    )

# Create context with shared components
context = Context(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)

# Create engine and add callbacks
trainer = Engine(train_step, context)
trainer.callbacks.append(MetricsCallback(metrics=[MultiClassAccuracy()]))

# Run training loop
trainer.loop(train_loader, max_epochs=10)
```