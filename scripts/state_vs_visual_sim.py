import subprocess as sp
import time

# num_envs_lst = [400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]
# num_envs_lst = [1,2,4,8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
num_envs_lst = [1,2,4,8, 16, 32, 64, 128, 256, 512]
seeds = [0, 1, 2, 3, 4]

for num_envs in num_envs_lst:
    for camera in ["state", "visual"]:
        for seed in seeds:

            if camera == "state":
                task = 'Isaac-Lift-Cube-Franka-v0'
            else:
                task = 'Isaac-Lift-Cube-Camera-Franka-v0'

            # ./orbit.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 2 --seed 0
            command = f"./orbit.sh -p source/standalone/workflows/skrl/train.py --task {task} --headless --num_envs {num_envs} --seed {seed}"

            start = time.time()
            result = sp.run(
                command.split(),
                stderr=sp.PIPE,
            )

            if result.returncode:
                print(f"Failed on {num_envs} envs")
                break

            # Save the time taken
            with open("time_state_vs_visual_sim.txt", "a") as f:
                # num_envs, seed, camera, time, full command
                f.writelines([f"{num_envs}, {seed}, {camera}, {time.time() - start}, {command}\n"])