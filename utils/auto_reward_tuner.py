import json
import os
import time
import shutil
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CONFIG_PATH = "f:/RLearning/config.py"
LOG_DIR = "runs"
BACKUP_PATH = "f:/RLearning/config_backup.py"

# 需要自动调参的奖励项及调整范围
REWARD_PARAMS = {
    'PENALTY_STATIC': [-5, -10, -20],
    'PENALTY_REPEAT_TRAJ': [-5, -10, -20],
    'REWARD_APPROACH_MAX': [4, 8, 12],
    'PENALTY_AWAY_SCALE': [3, 6, 10],
    'REWARD_DISTANCE': [1, 2, 4],
    'REWARD_SPEED_MAX': [2, 4, 8],
}

# 评估指标：目标靠近率、平均距离、平均reward
TARGET_METRIC = 'RewardItems/approach_reward'


def backup_config():
    shutil.copy(CONFIG_PATH, BACKUP_PATH)

def restore_config():
    shutil.copy(BACKUP_PATH, CONFIG_PATH)

def modify_config(param, value):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().startswith(param):
                f.write(f"{param} = {value}\n")
            else:
                f.write(line)

def read_tensorboard_metric(log_dir, tag, min_steps=100):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        return None
    events = event_acc.Scalars(tag)
    if len(events) < min_steps:
        return None
    vals = [e.value for e in events[-min_steps:]]
    return np.mean(vals)

def auto_tune():
    backup_config()
    best_score = -np.inf
    best_params = {}
    for param, values in REWARD_PARAMS.items():
        for v in values:
            print(f"[AutoTune] 尝试: {param} = {v}")
            modify_config(param, v)
            # 启动训练（假设main.py --mode train）
            os.system("python main.py --mode train --max_steps=200")
            # 等待训练产生日志
            time.sleep(10)
            metric = read_tensorboard_metric(LOG_DIR, TARGET_METRIC)
            if metric is not None and metric > best_score:
                best_score = metric
                best_params[param] = v
            print(f"[AutoTune] {param}={v}，approach_reward均值={metric}")
    print(f"[AutoTune] 最优参数: {best_params}, 最优approach_reward均值: {best_score}")
    restore_config()
    for param, v in best_params.items():
        modify_config(param, v)
    print("[AutoTune] 参数调整完毕！")

if __name__ == "__main__":
    auto_tune()
