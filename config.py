# 全局配置文件，集中管理超参数和奖励机制
CONFIG = {
    # 训练参数
    'MAX_EPISODES': 2000,
    'MAX_STEPS': 1000,
    'EPSILON_START': 1.0,
    'EPSILON_END': 0.01,
    'EPSILON_DECAY': 0.995,
    'GAMMA': 0.98,
    'LR': 5e-4,
    'BATCH_SIZE': 64,
    'REPLAY_BUFFER_SIZE': 1000000,
    'STACK_SIZE': 4,
    'MODEL_PATH': './model/lnn_agent.pth',
    'REWARD_SPEED_MAX': 2.0,  # 最大速度奖励
    'EXPLORATION_STEPS': 40000,  # 前4万步强制全探索（更大探索池）
    'MIN_SUCCESS_REWARD': 50.0,  # 达标奖励阈值，reward低于此值时到达目标点不会保存模型而是重置目标点

    # 学习率调度器参数
    'LR_SCHEDULER': {
        'type': 'ReduceLROnPlateau',  # 可选：'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR'
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-6,
        'mode': 'max',  # 以reward为指标
        'verbose': True
    },

    # 目标点偏移参数
    'TARGET_POINT_X_OFFSET': 2000.0,  # 目标点相对初始x偏移
    'TARGET_POINT_Y_OFFSET': 0.0,    # 目标点相对初始y偏移
    'TARGET_POINT_Z_OFFSET': 2.0,    # 目标点相对初始z偏移

    # 奖励机制（无人机专用，已更新）
    'REWARD_ALIVE': 0.1,         # 存活奖励
    'REWARD_DISTANCE': 3.0,      # 距离目标奖励（调低，防止静止时高reward）
    'REWARD_DISTANCE_MAX': 2000.0,  # 距离奖励归一化分母，必须与采样最大距离一致（与目标点采样逻辑匹配）
    'REWARD_SPEED_TARGET': 5.0,  # 目标速度（适中速度）
    'REWARD_SPEED_NORM': 5.0,  # 速度归一化分母，需与环境最大速度一致
    'REWARD_SPEED_SCALE': 0.2,   # 速度偏离惩罚系数
    'REWARD_Z_TARGET': -2.0,     # 理想高度
    'REWARD_Z_SCALE': 0.1,       # 高度偏离惩罚系数
    'REWARD_ATTITUDE_SCALE': 0.05, # 姿态偏离惩罚系数
    'PENALTY_COLLISION': -10.0,   # 碰撞惩罚
    'PENALTY_OFFROAD': -2.0,     # 偏离航线
    'PENALTY_SLOW': -20.0,        # 速度过慢惩罚（大幅提升）
    'PENALTY_STATIC': -40.0,      # 连续静止惩罚（大幅提升）
    'PENALTY_REPEAT_TRAJ': -2.0, # 震荡/小圈惩罚
    'PENALTY_ACTION_SWITCH': -1.0, # 动作切换频繁惩罚
    'REWARD_APPROACH_MAX': 10.0,    # 靠近目标点方向速度最高奖励（大幅提升）
    'REWARD_APPROACH_MAX_SPEED': 5.0,  # 靠近目标奖励归一化分母，建议与最大速度一致
    'PENALTY_AWAY_SCALE': 10.0,     # 远离目标点方向速度惩罚（大幅提升）

    # 轨迹检测参数
    'STATIC_STEPS': 10,          # 连续静止步数阈值
    'REPEAT_TRAJ_WINDOW': 20,    # 轨迹重复检测窗口
    'COLLISION_REPEAT_WINDOW': 10, # 碰撞重复检测窗口
    'ACTION_SWITCH_WINDOW': 5,   # 动作切换检测窗口
    'MIN_TOTAL_DISTANCE': 0.5,   # 静止判定最小累计距离

    # 推理参数
    'INFER_RENDER': True,
}
