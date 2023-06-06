# ml-tools

A collection of tools for monitoring, managing state and configuring machine learning experiments.

Inspired by and/or heavily based on:

- [CommonLoopUtils](https://github.com/google/CommonLoopUtils/)
- [ml_collections](https://github.com/google/ml_collections)

Written for JAX with minimal other dependencies.


## Highlight of features

### Dataclasses as configuration

Use dataclass to define your configuration. This allows you to use type hints and default values.

Use `setup_config` to parse command line arguments and override the default values.

Allows for nested dataclasses.

Example use. Assume a file `train.py` with the following content:
```python
from dataclasses import dataclass

from ml_tools.config_utils import setup_config

class OptimizationConfig:
    learning_rate: float = 1e-3
    momentum: float = 0.9

@dataclass
class Config:
    num_steps: int = 1024
    batch_size: int = 128
    num_train_examples: int = 50000
    optimizer: OptimizationConfig = OptimizationConfig()

    @property
    def num_epochs(self) -> int:
        return self.num_steps * self.batch_size // num_train_examples


config: Config = setup_config(Config)
print("num steps", config.num_steps)
```

Configure from the command line:
```bash
python train.py --config.num_steps=2048 --config.optimizer.learning_rate=1e-4
```


### Writers (for logging)

`ml-tools` provides `writes` for logging metrics, figures and configuration. Currently, the following writers are supported:
- Local: writes to disk.
Requires separate installation:
 - Tensorboard: `TensorBoardWriter`, using [`tensorboardX`](https://github.com/lanpa/tensorboardX).
 - Aim: `AimWriter`, using [`aim`](https://aimstack.io/). Aim is a local and open source alternative to Weights and Biases.

 It it possible to use multiple writers at the same time. This is useful for example to log to Tensorboard and Aim at the same time. This can be done using `MultiWriter`.

```python
from pathlib import Path
from ml_tools import writers
from dataclasses import asdict

experiment_name = "my_experiment"
experiment_dir = Path("logs")

# create writers
local_writer = writers.LocalWriter(str(experiment_dir), flush_every_n=100)
tb_writer = writers.TensorBoardWriter(str(expriment_dir / "tensorboard"))
aim_writer = writers.AimWriter(experiment_name)

# create a single writer dispatching to all writers
writer = writers.MultiWriter([aim_writer, tb_writer, local_writer])

# log config
writer.log_hparams(asdict(config))

# log metrics
writer.write_scalars(step=1, {"loss": 0.1, "accuracy": 0.9})
```


### Actions

Actions allow you to perform certain actions at certain steps. For example, you can use actions to save checkpoints, log metrics, or run evaluation.

Actions are defined by a callback function `callback_fn` which is called at certain steps. The following arguments are passed to the callback:
- `step`: the current step
- `t`: the current time
- `**kwargs`: additional arguments passed to the action. The callback can access these arguments by name, as illustrated in the example below:

```
from ml_tools import actions

def callback_action1(step, t, **kwargs):
    print("action1", step, t, kwargs["state"])

action1 = actions.PeriodicCallback(every_steps=10, callback_fn=callback_action1)

num_steps = 100
for i in range(num_steps):
    action1(step=i, t=None, state="state1")
```


### Checkpointing


Saving states to disk is supported using `save_checkpoint` and `load_checkpoint`. The state can by any `pytree`. We use `equinox` for serialization.

To save the state:
```python
from ml_tools import state_utils
state = {
    "weights": {"layer1": 1.0, "layer2": 2.0},
    "best_weights": {"layer1": 0.0, "layer2": 0.0},
}

state_utils.save_checkpoint(state, directory="checkpoints", step=100)
```

Loading it typically happens in two steps:
- find the latest checkpoint step
- load the checkpoint

```python
index = state_utils.find_latest_checkpoint_step_index(checkpoint_dir)
if index is not None:
    state = state_utils.load_checkpoint(state, checkpoint_dir, step_index=index)
```

Note that one needs access to the state's pytree to load the checkpoint.
