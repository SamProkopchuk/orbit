

# Orbit

## Installation

Follow:
- https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html
- https://isaac-orbit.github.io/orbit/source/setup/installation.html

```bash
./orbit.sh --extra skrl
```

```bash
sudo reboot
```

### Test Orbit is working
```bash
# Option 1: Using the orbit.sh executable
./orbit.sh -p source/standalone/tutorials/00_sim/create_empty.py --headless
# Option 2: Using python in your virtual environment
python source/standalone/tutorials/00_sim/create_empty.py
```

### Run a standard task
```bash
# optionally add --usewandb, --timesteps 100, ...
./orbit.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 2 --seed 0
./orbit.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Camera-Franka-v0 --headless --num_envs 2 --seed 0
```

### Orbit visual vs state-based profiling

```bash
mamba activate orbit
python scripts/state_vs_visual_sim/state_vs_visual_sim.py
```

## Important files 

- `source/standalone/workflows/skrl/train.py` - The main training script for the SKRL agent.
- `source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/lift/config/franka/joint_pos_env_camera_cfg.py` - The configuration file for the Isaac Gym environment, where we define the task.
- `isaac.orbit_tasks/omni/isaac/orbit_tasks/utils/wrappers/skrl.py` - The SKRL wrapper for the Isaac Gym environment. If you want to change the input / output to the agent (e.g. pass images).
- `source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/lift/config/franka/agents/skrl_ppo_cfg.yaml` - The configuration file for the SKRL agent.

If you want to add a new task:
- Copy `source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/lift/config/franka/joint_pos_env_cfg.py`
- Modify the task to your needs
- Register the task like in `source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/lift/config/franka/__init__.py`