# timeBasedRIS

Time-based Spiking Neural Network for RIS passive beamforming configuration.

## Project structure

- `src/neuron/`     : Morris-Lecar neuron model and TTFS encoding
- `src/network/`    : SNN architecture for RIS configuration
- `src/channel/`    : Rician+ULA channel model
- `src/evaluation/` : Metrics and energy analysis
- `config/`         : Hyperparameters
- `scripts/`        : Experiment entry points
- `notebooks/`      : Exploration and figures
- `tests/`          : Unit tests

## Setup

    uv sync

## References

- Sourikopoulos et al., A 4-fJ/Spike Artificial Neuron in 65nm CMOS, Frontiers in Neuroscience, 2017
- Morris and Lecar, Voltage oscillations in the barnacle giant muscle fiber, 1981
