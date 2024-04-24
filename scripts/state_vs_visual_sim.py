import subprocess as sp
import time
import wandb
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(parent_dir, "time_state_vs_visual_sim.txt")

# num_envs_lst = [400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]
# num_envs_lst = [1,2,4,8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
num_envs_lst = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    for num_envs in num_envs_lst:
        for camera in ["state", "visual"]:

            print()
            print('-'* 80)
            print()

            if camera == "state":
                task = 'Isaac-Lift-Cube-Franka-v0'
            else:
                task = 'Isaac-Lift-Cube-Camera-Franka-v0'

            # ./orbit.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 2 --seed 0
            command = f"./orbit.sh -p source/standalone/workflows/skrl/train.py --task {task} --headless --num_envs {num_envs} --seed {seed} --usewandb"
            
            # run = wandb.init(project="orbit", group='state_vs_visual_sim', name=f"{camera}-n{num_envs}-s{seed}")
            # print(f'run_id: {run.id}')
            # wandb.config.update({"camera": camera, "num_envs": num_envs, "seed_sim": seed, "task": task, "command": command})
            # wandb.log({'i': 0}) # dummy to get wandb to log system metrics

            start = time.time()
            result = sp.run(
                command.split(),
                stderr=sp.PIPE,
            )
            runtime = time.time() - start

            # wandb.summary.update({"runtime": runtime})
            # wandb.finish()

            if result.returncode:
                print(f"Failed on {num_envs} envs")
                break

            # Save the time taken
            with open(file_dir, "a") as f:
                # num_envs, seed, camera, time, full command
                f.writelines([f"{num_envs}, {seed}, {camera}, {runtime}, {command}\n"])