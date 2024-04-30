
# Profilers

```bash
mamba activate orbit
# optionally add --usewandb, --timesteps 100, ...
./orbit.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000

```

## cProfile
```bash
# python -m cProfile -o output.pstats <your_script.py> arg1 arg2 …
python -m cProfile -o statebased128.pstats source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000
python -m cProfile -o visual128.pstats source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Camera-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000

pip install graphviz
pip install gprof2dot
gprof2dot -f pstats output.pstats | dot -Tpng -o output.png
```

## Robert Kern line_profiler

### With decorator
in python file:
```python
import line_profiler 
@line_profiler.profile
```

```bash
pip install --no-cache-dir --upgrade --force-reinstall line-profiler>=4.1.3
# legacy: kernprof -l -v -o statebased128.lprof main.py

LINE_PROFILE=1 source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000

python -m kernprof -l -v -o statebased128.lprof source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000
python -m kernprof -l -v -o visual128.lprof source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Camera-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000
```

### Better: autoprofiler

```bash
python -m kernprof -l -v -o statebased128.lprof -p source/standalone/workflows/skrl/train.py source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000
```

```bash
snakeviz program.prof
```

## Nsight Systems

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -o my_profile python source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Camera-Franka-v0 --headless --num_envs 128 --seed 0 --timesteps 1000
```

* open nsight systems app
* drag and drop the .nsys-rep file

---

# Analysis

Stack trace:
```python
# source/standalone/workflows/skrl/train.py
main():
    gym.make()

    # source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/base_env.py
        BaseEnv.__init__():
            self.sim.reset()

                # isaac_sim-2023.1.1/exts/omni.isaac.core/omni/isaac/core/simulation_context/simulation_context.py
                SimulationContext.step()
                    # isaac_sim-2023.1.1/kit/kernel/py/omni/kit/app/_impl/app_iface.py
                    # self._app = omni.kit.app.get_app_interface()
                    self._app.update() # this takes exponentially long in num_envs
```
Note: `~/.local/share/ov/pkg/isaac_sim-2023.1.1/` is the default path to the installed Isaac Sim.