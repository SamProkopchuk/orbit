"""
Loop over all logged runs and add a metric to each run.
When we used `run.summary[] = value`, we were not able to use the metric in the wandb UI plots.
So load the values from the history, and log them using `wandb.log()`.
https://github.com/wandb/wandb/issues/3547
"""
import wandb
import tqdm
import numpy as np

ENTITY = "andreas-burger"
PROJECT = "orbit"

# Metrics to add
ALL_METRIC_NAMES = [ 
    # 'system/proc.memory.rssMB',
    'system.proc.memory.rssMB',
    # 'system/gpu.process.0.memoryAllocatedBytes',
    'system.gpu.process.0.memoryAllocatedBytes',
]

api = wandb.Api()
runs = api.runs(path=f"{ENTITY}/{PROJECT}") # can also filter here as usual


for run in tqdm.tqdm(runs):
    
    run_id = run.id
    run_again = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume='must' # This is important!
    )
    # history = run.history()
    history = run.history(stream="system")

    for metric in ALL_METRIC_NAMES:
        if metric not in history.columns:
            continue

        values = history[metric].values
        values = values[~np.isnan(values)]

        # -> Won't be able to use it in wandb UI plots
        # run.summary[f"total_{metric}"] = np.mean(values) 
        # run.summary[f"total_{metric}_std"] = np.std(values)

        # -> Plotting works, because summary is updated during the run
        # wandb.log({f"total_{metric}": np.mean(values)})) 
        # wandb.log({f"total_{metric}_std": np.std(values)}))  

        print(f"max_{metric}: {np.max(values)}")
        wandb.log({f"max_{metric}": np.max(values)})

    # run.summary.update() #-> Needed for updating the summary, don't need if log using wandb.log()
    wandb.finish()

