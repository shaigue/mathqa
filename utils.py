import json

import config


def get_experiment_logs(experiment_name: str):
    filename = 'train_log.json'
    experiment_dir = config.TRAINING_LOGS_DIR / experiment_name / filename
    with experiment_dir.open('r') as f:
        experiment_logs = json.load(f)
    return experiment_logs