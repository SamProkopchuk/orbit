

# Orbit

## Installation

Follow:
https://isaac-orbit.github.io/orbit/source/setup/installation.html

```bash
./orbit.sh --extra skrl
```

### Test Orbit is working
```bash
# Option 1: Using the orbit.sh executable
./orbit.sh -p source/standalone/tutorials/00_sim/create_empty.py
# Option 2: Using python in your virtual environment
python source/standalone/tutorials/00_sim/create_empty.py
```

### Run a standard task
```bash
./orbit.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 2 --seed 0
```

