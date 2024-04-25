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

# System metrics were we want to log the maximum value
SYSTEM_METRIC_LOG_MAX = [ 
    # 'system/proc.memory.rssMB',
    'system.proc.memory.rssMB',
    # 'system/gpu.process.0.memoryAllocatedBytes',
    'system.gpu.process.0.memoryAllocatedBytes',
]

# A value we want to log from the summary
METRIC_LOG_SUMMARY = [
    'totaltime',
    'traintime',
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
    history = run.history(samples=10000)
    syshistory = run.history(samples=10000, stream="system") # system, systemMetrics
    # scanhistory = run.scan_history() # system, systemMetrics
    # values = [r[metric] for r in syshistory if metric in r]
    step = wandb.run.step

    print(f"Run: {run_id}")

    for metric in SYSTEM_METRIC_LOG_MAX:
        if metric not in syshistory.columns:
            print(f"{metric} not in syshistory. Skipping...")
            continue
        if f'max_{metric}' in history:
            print(f"max_{metric} already in history. Skipping...")
            continue

        values = syshistory[metric].values
        values = values[~np.isnan(values)]

        # -> Won't be able to use it in wandb UI plots
        # run.summary[f"total_{metric}"] = np.mean(values) 
        # run.summary[f"total_{metric}_std"] = np.std(values)

        # -> Plotting works, because summary is updated during the run
        # wandb.log({f"total_{metric}": np.mean(values)})) 
        # wandb.log({f"total_{metric}_std": np.std(values)}))  

        # wandb: WARNING Step only supports monotonically increasing values, use define_metric to set a custom x axis. For details see: https://wandb.me/define-metric
        # wandb: WARNING (User provided step: 0 is less than current step: 2. Dropping entry
        print(f"max_{metric}: {np.max(values)}")
        wandb.log({f"max_{metric}": np.max(values)}, step=step)
    
    for metric in METRIC_LOG_SUMMARY:
        if metric not in run.summary:
            print(f"{metric} not in summary. Skipping...")
            continue
        if f'{metric}' in history:
            print(f"{metric} already in history. Skipping...")
            continue

        value = run.summary[metric]

        print(f"{metric}: {value}")
        wandb.log({metric: value}, step=step)

    # run.summary.update() #-> Needed for updating the summary, don't need if log using wandb.log()
    wandb.finish()

