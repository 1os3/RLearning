import os
import torch
import numpy as np
from env.airsim_env import AirSimVisionEnv
from agent.vision_agent import VisionLNNAgent
from utils.vision_utils import preprocess_image, stack_state
from config import CONFIG
import random
import time
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from utils.amp_utils import autocast_xpu
import gc

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *transition):
        self.buffer.append(tuple(transition))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 兼容拓展：state, action, reward, next_state, delta_pos, target_pos, elapsed_time, lowdim, done
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

class RewardScheduler:
    def __init__(self, initial_config):
        self.config = initial_config.copy()
        self.history = []
        self.max_history = 1000
    def update_metrics(self, reward, approach_reward):
        self.history.append((reward, approach_reward))
        if len(self.history) > self.max_history:
            self.history.pop(0)
    def step(self):
        if len(self.history) < 100:
            return
        avg_reward = np.mean([r for r, _ in self.history])
        avg_approach = np.mean([a for _, a in self.history])
        # 策略1: 靠近目标奖励低则提升REWARD_APPROACH_MAX
        if avg_approach < 0.5:
            self.config['REWARD_APPROACH_MAX'] = min(self.config['REWARD_APPROACH_MAX'] + 1, 20)
        # 策略2: 总reward低则加大静止惩罚
        if avg_reward < 0:
            self.config['PENALTY_STATIC'] = min(self.config['PENALTY_STATIC'] - 1, -20)
    def get(self, key):
        return self.config.get(key)

def calc_reward(info, last_info=None, context=None, config=CONFIG, step_count=None):
    reward = config['REWARD_ALIVE']
    # 距离目标点奖励（归一化）
    distance = float(info.get('distance', 1))
    # === Reward相关debug输出暂时注释 ===
    # print('[DEBUG][reward] info[x]:', type(info.get('x', 0)), info.get('x', 0))
    # print('[DEBUG][reward] last_info[x]:', type(last_info.get('x', 0)) if last_info else None, last_info.get('x', 0) if last_info else None)
    # print('[DEBUG][reward] info[y]:', type(info.get('y', 0)), info.get('y', 0))
    # print('[DEBUG][reward] last_info[y]:', type(last_info.get('y', 0)) if last_info else None, last_info.get('y', 0) if last_info else None)
    # print('[DEBUG][reward] info[z]:', type(info.get('z', 0)), info.get('z', 0))
    # print('[DEBUG][reward] last_info[z]:', type(last_info.get('z', 0)) if last_info else None, last_info.get('z', 0) if last_info else None)
    reward += config['REWARD_DISTANCE'] * max(0, config['REWARD_DISTANCE_MAX'] - distance) / config['REWARD_DISTANCE_MAX']
    # 速度偏离惩罚
    speed = float(info.get('speed', 0))
    if speed < config['REWARD_SPEED_TARGET']:
        reward -= abs(speed - config['REWARD_SPEED_TARGET']) * config['REWARD_SPEED_SCALE']
    # 高度偏离惩罚
    z = float(info.get('z', config['REWARD_Z_TARGET']))
    reward -= abs(z - config['REWARD_Z_TARGET']) * config['REWARD_Z_SCALE']
    # 姿态偏离惩罚
    pitch = abs(float(info.get('pitch', 0)))
    roll = abs(float(info.get('roll', 0)))
    reward -= (pitch + roll) * config['REWARD_ATTITUDE_SCALE']
    # 碰撞惩罚
    if info.get('collision', False):
        reward += config['PENALTY_COLLISION']
    # 偏离航线惩罚
    if info.get('offroad', False):
        reward += config['PENALTY_OFFROAD']
    # 速度过慢惩罚
    if speed < 0.1:
        reward += config['PENALTY_SLOW']
    # === 新增：鼓励持续运动 ===
    if abs(info.get('speed', 0)) > 0.05:
        reward += 0.1  # 鼓励有位移
    # === 静止惩罚 ===
    if abs(info.get('speed', 0)) < 0.05:
        reward += config.get('PENALTY_STATIC', -5.0)
    # 震荡/小圈惩罚
    if context is not None and 'repeat_traj' in context and context['repeat_traj']:
        reward += config.get('PENALTY_REPEAT_TRAJ', -10.0)
    # 动作切换频繁惩罚
    if context is not None and 'action_switch' in context and context['action_switch']:
        reward += config.get('PENALTY_ACTION_SWITCH', -1.0)
    # 速度奖励：越快奖励越接近REWARD_SPEED_MAX，最高为REWARD_SPEED_MAX
    max_speed = config['REWARD_SPEED_NORM']  # 可根据动作空间调整
    speed_reward = min(speed, max_speed) / max_speed * config.get('REWARD_SPEED_MAX', 2.0)
    reward += speed_reward
    # 靠近目标点方向速度奖励/惩罚
    # 计算无人机指向目标点的单位向量与速度矢量投影
    if last_info is not None and context is not None and 'target_pos' in context:
        pos = np.array([float(info.get('x', 0)), float(info.get('y', 0)), float(info.get('z', 0))])
        last_pos = np.array([float(last_info.get('x', 0)), float(last_info.get('y', 0)), float(last_info.get('z', 0))])
        tgt = np.array([float(v) for v in context['target_pos']])
        to_target_vec = tgt - pos
        to_target_unit = to_target_vec / (np.linalg.norm(to_target_vec) + 1e-8)
        v_vec = pos - last_pos
        v_proj = np.dot(v_vec, to_target_unit)
        max_proj_speed = config['REWARD_APPROACH_MAX_SPEED']
        if v_proj > 0:
            # 靠近目标点，奖励线性递增，最高2分
            approach_reward = min(v_proj, max_proj_speed) / max_proj_speed * config.get('REWARD_APPROACH_MAX', 8.0)
        else:
            # 远离目标点，惩罚与远离速度线性相关，无上限
            approach_reward = v_proj * config.get('PENALTY_AWAY_SCALE',6.0)
        reward += approach_reward
    # === 5min内未移动超过10m惩罚 ===
    if 'pos_history' not in context:
        context['pos_history'] = []
    if 'time_history' not in context:
        context['time_history'] = []
    cur_pos = np.array([float(info.get('x', 0)), float(info.get('y', 0)), float(info.get('z', 0))], dtype=np.float32)
    cur_time = time.time()
    context['pos_history'].append(cur_pos)
    context['time_history'].append(cur_time)
    # 只保留最近5分钟数据
    while context['time_history'] and cur_time - context['time_history'][0] > 300:
        context['time_history'].pop(0)
        context['pos_history'].pop(0)
    if context['time_history'] and cur_time - context['time_history'][0] >= 300:
        dist = np.linalg.norm(cur_pos - context['pos_history'][0])
        if dist < 10.0:
            reward += config.get('PENALTY_NO_MOVE', -20.0)
            if step_count is not None and step_count % 100 == 0:
                print(f'[惩罚] 5分钟内未移动超过10m，距离={dist:.2f}，已施加惩罚')
    return reward

def check_static(speeds, config):
    # 检查最近STATIC_STEPS步内是否静止
    return all(s < 0.05 for s in speeds[-config['STATIC_STEPS']:]) if len(speeds) >= config['STATIC_STEPS'] else False

def check_repeat_traj(positions, config):
    # 检查最近REPEAT_TRAJ_WINDOW步内是否有位置重复（小圈）
    if len(positions) < config['REPEAT_TRAJ_WINDOW']:
        return False
    recent = positions[-config['REPEAT_TRAJ_WINDOW']:]
    arr = np.array(recent)
    dists = np.linalg.norm(arr - arr[-1], axis=1)
    return np.sum(dists < 0.5) > 3  # 0.5米内重复超过3次

def check_action_switch(actions, config):
    if len(actions) < config['ACTION_SWITCH_WINDOW']:
        return False
    recent = actions[-config['ACTION_SWITCH_WINDOW']:]
    return len(set(recent)) > 2

def check_collision_repeat(collisions, config):
    # 检查最近COLLISION_REPEAT_WINDOW步内碰撞次数
    return sum(collisions[-config['COLLISION_REPEAT_WINDOW']:]) > 1

def get_scheduler(optimizer):
    sched_conf = CONFIG['LR_SCHEDULER']
    if sched_conf['type'] == 'ReduceLROnPlateau':
        # verbose参数不支持，需移除
        return ReduceLROnPlateau(optimizer, factor=sched_conf['factor'], patience=sched_conf['patience'], min_lr=sched_conf['min_lr'], mode=sched_conf['mode'])
    elif sched_conf['type'] == 'StepLR':
        return StepLR(optimizer, step_size=sched_conf.get('step_size', 20), gamma=sched_conf.get('factor', 0.5))
    elif sched_conf['type'] == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=sched_conf.get('T_max', 100), eta_min=sched_conf.get('min_lr', 1e-6))
    else:
        return None

OPEN_AREA_BOUNDS = {
    'x': (-25, 25),
    'y': (-25, 25),
    'z': (-4, -2)
}
OBSTACLE_LIST = [  # 障碍物中心点和半径，实际可根据地图自动生成
    {'center': [0, 0, -2], 'radius': 5},
    {'center': [10, 10, -2], 'radius': 5},
    # ...如有更多障碍物，可补充
]

# 判断点是否在障碍物范围内
def is_in_obstacle(pos, obstacle_list):
    for obs in obstacle_list:
        if np.linalg.norm(np.array(pos) - np.array(obs['center'])) < obs['radius']:
            return True
    return False

# 随机采样一个合格的目标点
def sample_valid_target(drone_pos, min_dist=10.0, max_trials=100):
    for _ in range(max_trials):
        x = random.uniform(*OPEN_AREA_BOUNDS['x'])
        y = random.uniform(*OPEN_AREA_BOUNDS['y'])
        z = random.uniform(*OPEN_AREA_BOUNDS['z'])
        tgt = [x, y, z]
        if np.linalg.norm(np.array(tgt) - np.array(drone_pos)) >= min_dist and not is_in_obstacle(tgt, OBSTACLE_LIST):
            return tgt
    # fallback:直接远离无人机
    return [drone_pos[0]+min_dist, drone_pos[1], drone_pos[2]]

def sample_target_in_open_area(env, drone_pos, min_dist=10.0, max_dist=20.0, max_trials=100):
    open_points = env.get_navigable_points()
    np.random.shuffle(open_points)
    for tgt in open_points:
        dist = np.linalg.norm(np.array(tgt) - np.array(drone_pos))
        if min_dist <= dist <= max_dist:
            return tgt
    # fallback: 直接在无人机前方固定距离
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    return (np.array(drone_pos) + direction * min_dist).tolist()

def build_reward_items(info, last_info, context, reward_scheduler, CONFIG):
    reward_items = {}
    distance = float(info.get('distance', 1))
    speed = float(info.get('speed', 0))
    z = float(info.get('z', CONFIG['REWARD_Z_TARGET']))
    pitch = abs(float(info.get('pitch', 0)))
    roll = abs(float(info.get('roll', 0)))
    reward_items['alive'] = reward_scheduler.config['REWARD_ALIVE']
    reward_items['distance'] = reward_scheduler.config['REWARD_DISTANCE'] * max(0, CONFIG['REWARD_DISTANCE_MAX'] - distance) / CONFIG['REWARD_DISTANCE_MAX']
    reward_items['speed_penalty'] = -abs(speed - reward_scheduler.config['REWARD_SPEED_TARGET']) * reward_scheduler.config['REWARD_SPEED_SCALE'] if speed < reward_scheduler.config['REWARD_SPEED_TARGET'] else 0.0
    reward_items['z_penalty'] = -abs(z - reward_scheduler.config['REWARD_Z_TARGET']) * reward_scheduler.config['REWARD_Z_SCALE']
    reward_items['attitude_penalty'] = -(pitch + roll) * reward_scheduler.config['REWARD_ATTITUDE_SCALE']
    reward_items['collision'] = reward_scheduler.config['PENALTY_COLLISION'] if info.get('collision', False) else 0.0
    reward_items['offroad'] = reward_scheduler.config['PENALTY_OFFROAD'] if info.get('offroad', False) else 0.0
    reward_items['slow'] = reward_scheduler.config['PENALTY_SLOW'] if speed < 0.1 else 0.0
    reward_items['static'] = reward_scheduler.config['PENALTY_STATIC'] if context and 'static_steps' in context and context['static_steps'] >= 10 else 0.0
    reward_items['repeat_traj'] = reward_scheduler.config.get('PENALTY_REPEAT_TRAJ', -10.0) if context is not None and 'repeat_traj' in context and context['repeat_traj'] else 0.0
    reward_items['action_switch'] = reward_scheduler.config.get('PENALTY_ACTION_SWITCH', -1.0) if context is not None and 'action_switch' in context and context['action_switch'] else 0.0
    max_speed = CONFIG['REWARD_SPEED_NORM']
    reward_items['speed_reward'] = min(speed, max_speed) / max_speed * reward_scheduler.config.get('REWARD_SPEED_MAX', 2.0)
    # 靠近目标奖励
    if last_info is not None and context is not None and 'target_pos' in context:
        pos = np.array([float(info.get('x', 0)), float(info.get('y', 0)), float(info.get('z', 0))])
        last_pos = np.array([float(last_info.get('x', 0)), float(last_info.get('y', 0)), float(last_info.get('z', 0))])
        tgt = np.array([float(v) for v in context['target_pos']])
        to_target_vec = tgt - pos
        to_target_unit = to_target_vec / (np.linalg.norm(to_target_vec) + 1e-8)
        v_vec = pos - last_pos
        v_proj = np.dot(v_vec, to_target_unit)
        max_proj_speed = CONFIG['REWARD_APPROACH_MAX_SPEED']
        if v_proj > 0:
            reward_items['approach_reward'] = min(v_proj, max_proj_speed) / max_proj_speed * reward_scheduler.config.get('REWARD_APPROACH_MAX', 8.0)
        else:
            reward_items['approach_reward'] = v_proj * reward_scheduler.config.get('PENALTY_AWAY_SCALE', 6.0)
    else:
        reward_items['approach_reward'] = 0.0
    return reward_items

def write_reward_items(writer, reward_items, step_count):
    for k, v in reward_items.items():
        writer.add_scalar(f'RewardItems/{k}', v, step_count)

def log_warnings(info, ep, step_count):
    if info.get('collision', False):
        print(f'[警告] 第{ep+1}集第{step_count}步发生碰撞')
    if info.get('offroad', False):
        print(f'[警告] 第{ep+1}集第{step_count}步偏离航线')

def train():
    # 设置随机种子以保证可复现
    seed = CONFIG.get('SEED', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = AirSimVisionEnv()
    # 设备配置，优先xpu，其次cuda，最后cpu
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # 采样一组stack_state，推断真实shape
    img = env.get_image()
    state = preprocess_image(img)
    state_list = [state for _ in range(CONFIG['STACK_SIZE'])]
    stacked_state = stack_state(state_list)
    print('stacked_state.shape:', stacked_state.shape)
    # 低维状态shape推断
    sample_info = env.get_info() if hasattr(env, 'get_info') else {}
    lowdim_keys = ['speed','x','y','z','pitch','roll','yaw']
    lowdim_dim = len([k for k in lowdim_keys if k in sample_info]) if sample_info else 9
    agent = VisionLNNAgent(input_shape=stacked_state.shape, action_dim=3, lowdim_dim=lowdim_dim, device=device)
    replay = ReplayBuffer(CONFIG['REPLAY_BUFFER_SIZE'])
    epsilon = CONFIG['EPSILON_START']
    gamma = CONFIG['GAMMA']
    best_reward = -float('inf')
    start_ep = 0
    if os.path.exists(CONFIG['MODEL_PATH']):
        agent.load(CONFIG['MODEL_PATH'])
        print(f"[Train] 加载断点模型: {CONFIG['MODEL_PATH']}")
    os.makedirs(os.path.dirname(CONFIG['MODEL_PATH']), exist_ok=True)
    scheduler = get_scheduler(agent.optimizer)
    writer = SummaryWriter(log_dir="runs/lnn_rl")

    reward_scheduler = RewardScheduler(CONFIG.copy())

    for ep in trange(start_ep, CONFIG['MAX_EPISODES'], desc="Episode"):
        try:
            env.reset()
            agent.reset_state()
            state_list = []
            img = env.get_image()
            state = preprocess_image(img)
            for _ in range(CONFIG['STACK_SIZE']):
                state_list.append(state)
            stacked_state = stack_state(state_list)
            total_reward = 0
            done = False
            last_info = None
            losses = []
            # 目标点：episode开始时唯一确定
            drone_init_info = env.get_info()
            drone_init_pos = [float(drone_init_info['x']), float(drone_init_info['y']), float(drone_init_info['z'])]
            target_pos = [drone_init_pos[0] + 2000.0, drone_init_pos[1], drone_init_pos[2] + 2.0]
            context = {
                'last_distance': 1.0,
                'total_distance': 0.0,
                'static_steps': 0,
                'repeat_traj': False,
                'collision_count': 0,
                'action_switch': False,
                'init_pos': drone_init_pos,
                'start_time': time.time(),
                'target_pos': target_pos
            }
            speeds, positions, actions, collisions = [], [], [], []
            step_count = 0
            while True:
                # === 1. 采集低维状态 ===
                info = env.get_info(target_pos=context['target_pos']) if hasattr(env, 'get_info') else {}
                # === DEBUG: 打印所有target_pos类型和值 ===
                #print('[DEBUG] context[target_pos] type:', type(context['target_pos']), context['target_pos'])
                #print('[DEBUG] info[target_pos] type:', type(info.get('target_pos', None)), info.get('target_pos', None))
                # === 2. 低维状态 ===
                lowdim_keys = ['speed','x','y','z','pitch','roll','yaw']
                lowdim = np.array([float(info.get(k,0)) for k in lowdim_keys], dtype=np.float32)
                lowdim_tensor = torch.from_numpy(lowdim).unsqueeze(0)  # (1, lowdim_dim)
                # === 新增：相对初始点位移、目标点坐标、已用时间 ===
                try:
                    cur_pos = np.array([
                        float(info.get('x', 0)),
                        float(info.get('y', 0)),
                        float(info.get('z', 0))
                    ], dtype=np.float32)
                    delta_pos = cur_pos - np.array(context['init_pos'], dtype=np.float32)
                except Exception as e:
                    print('[ERROR] delta_pos异常:', type(cur_pos), cur_pos, type(context['init_pos']), context['init_pos'], e)
                    raise
                try:
                    delta_pos_tensor = torch.from_numpy(delta_pos).unsqueeze(0)
                    target_pos_tensor = torch.from_numpy(np.array(context['target_pos'], dtype=np.float32)).unsqueeze(0)
                except Exception as e:
                    print('[ERROR] target_pos_tensor异常:', type(context['target_pos']), context['target_pos'], e)
                    raise
                elapsed_time = np.array([time.time() - context['start_time']], dtype=np.float32)
                elapsed_time_tensor = torch.from_numpy(elapsed_time).unsqueeze(0)
                # === 3. 状态堆叠 ===
                img = env.get_image()
                state = preprocess_image(img)
                state_list.pop(0)
                state_list.append(state)
                next_stacked_state = stack_state(state_list)
                # === 4. 模型推理 ===
                with autocast_xpu():
                    out, _ = agent.forward(
                        stacked_state.unsqueeze(0),
                        lowdim=lowdim_tensor,
                        delta_pos=delta_pos_tensor,
                        target_pos=target_pos_tensor,
                        elapsed_time=elapsed_time_tensor
                    )
                # === 增加传递给模型的数据debug输出 ===
                #print('[DEBUG][model input] stacked_state:', type(stacked_state), stacked_state.shape if hasattr(stacked_state, 'shape') else None)
                #print('[DEBUG][model input] lowdim:', type(lowdim), lowdim.shape if hasattr(lowdim, 'shape') else None)
                #print('[DEBUG][model input] delta_pos:', type(delta_pos), delta_pos.shape if hasattr(delta_pos, 'shape') else None)
                #print('[DEBUG][model input] target_pos:', type(target_pos), target_pos)
                #print('[DEBUG][model input] elapsed_time:', type(elapsed_time), elapsed_time.shape if hasattr(elapsed_time, 'shape') else None, elapsed_time)
                #print('[DEBUG][pos]', info.get('x'), info.get('y'), info.get('z'))
                action = agent.select_action(out, epsilon)
                vx, vy, vz = action if len(action) == 3 else (0.0, 0.0, 0.0)
                env.step(vx, vy, vz)
                info = env.get_info(target_pos=context['target_pos'])
                speeds.append(float(info['speed']))
                positions.append([
                    float(info['x']),
                    float(info['y']),
                    float(info['z'])
                ])
                actions.append(tuple(action))
                collisions.append(int(info['collision']))
                context['total_distance'] += float(info['speed'])
                context['static_steps'] = context['static_steps']+1 if float(info['speed']) < 0.05 else 0
                context['repeat_traj'] = check_repeat_traj(positions, CONFIG)
                context['action_switch'] = check_action_switch(actions, CONFIG)
                context['collision_count'] = sum(collisions[-CONFIG['COLLISION_REPEAT_WINDOW']:])
                if float(info['distance']) < 1.0:
                    total_reward += 20.0
                reward = calc_reward(info, last_info, context, config=reward_scheduler.config, step_count=step_count)
                approach_reward = context.get('approach_reward', 0) if context else 0
                reward_scheduler.update_metrics(reward, approach_reward)
                if step_count % 1000 == 0:
                    reward_scheduler.step()
                last_info = info
                action = np.asarray(action, dtype=np.float32)
                assert action.shape == (agent.action_dim,), f"action shape error: {action.shape}"
                # 经验回放拓展：存储全部新特征
                replay.push(
                    stacked_state.numpy(),
                    action,
                    reward,
                    next_stacked_state.numpy(),
                    delta_pos,
                    [float(v) for v in context['target_pos']],
                    elapsed_time,
                    lowdim,
                    False # done
                )
                total_reward += reward
                stacked_state = next_stacked_state
                step_count += 1
                if step_count % 100 == 0:
                    print(f"[Step {step_count}] Loss: {loss.item():.4f}, Reward: {reward:.2f}, Epsilon: {epsilon:.3f}, Speed: {float(info.get('speed', 0)):.2f}, Distance: {float(info.get('distance', 0)):.2f}, Collision: {info.get('collision', False)}")
                reward_items = build_reward_items(info, last_info, context, reward_scheduler, CONFIG)
                write_reward_items(writer, reward_items, step_count)
                log_warnings(info, ep, step_count)
                if step_count % 10 == 0:
                    writer.add_image('AI视觉输入', img.transpose(2, 0, 1), global_step=step_count, dataformats='CHW')
                if step_count % 2 == 0 and len(replay) >= CONFIG['BATCH_SIZE']:
                    try:
                        b_s, b_a, b_r, b_ns, b_delta_pos, b_target_pos, b_elapsed_time, b_lowdim, b_d = zip(*random.sample(replay.buffer, CONFIG['BATCH_SIZE']))
                    except Exception as e:
                        print('[ERROR] batch采样异常:', e)
                        raise
                    try:
                        b_target_pos_arr = np.array([list(map(float, x)) for x in b_target_pos], dtype=np.float32)
                    except Exception as e:
                        print('[ERROR] b_target_pos转float异常:', type(b_target_pos[0]), b_target_pos[0], e)
                        raise
                    try:
                        b_s = torch.tensor(np.array(b_s), dtype=torch.float32, device=device)
                        b_a = torch.tensor(np.array(b_a), dtype=torch.float32, device=device)
                        b_r = torch.tensor(b_r, dtype=torch.float32, device=device)
                        b_ns = torch.tensor(np.array(b_ns), dtype=torch.float32, device=device)
                        b_delta_pos = torch.tensor(np.array(b_delta_pos), dtype=torch.float32, device=device)
                        b_target_pos = torch.tensor(b_target_pos_arr, dtype=torch.float32, device=device)
                        b_elapsed_time = torch.tensor(np.array(b_elapsed_time), dtype=torch.float32, device=device)
                        b_lowdim = torch.tensor(np.array(b_lowdim), dtype=torch.float32, device=device)
                        b_d = torch.tensor(b_d, dtype=torch.float32, device=device)
                    except Exception as e:
                        print('[ERROR] batch张量转换异常:', e)
                        raise
                    with autocast_xpu():
                        with torch.no_grad():
                            out, _ = agent.forward(b_ns, b_a, lowdim=b_lowdim, delta_pos=b_delta_pos, target_pos=b_target_pos, elapsed_time=b_elapsed_time)
                            q_next = out["q"]
                            q_target = b_r + gamma * (1 - b_d) * q_next.squeeze()
                        loss = agent.compute_loss(b_s, b_a, q_target, lowdim=b_lowdim, delta_pos=b_delta_pos, target_pos=b_target_pos, elapsed_time=b_elapsed_time)
                        agent.update(loss)
                        losses.append(loss.detach().item())
                if step_count % 100 == 0:
                    agent.save(CONFIG['MODEL_PATH'])
                if done:
                    break
            avg_loss = np.mean(losses) if losses else 0.0
            lr = agent.optimizer.param_groups[0]['lr']
            print(f"Episode {ep+1}/{CONFIG['MAX_EPISODES']}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, LR: {lr:.6f}")
            writer.add_scalar('Reward/Episode', total_reward, ep+1)
            writer.add_scalar('Loss/Episode', avg_loss, ep+1)
            writer.add_scalar('Epsilon/Episode', epsilon, ep+1)
            writer.add_scalar('LearningRate/Episode', lr, ep+1)
            epsilon = max(CONFIG['EPSILON_END'], epsilon * CONFIG['EPSILON_DECAY'])
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(total_reward)
                else:
                    scheduler.step()
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(CONFIG['MODEL_PATH'])
            gc.collect()
            if hasattr(torch, 'xpu'):
                torch.xpu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Train] Episode {ep+1} 异常: {e}")
            continue
    # 训练结束后关闭环境、保存模型和关闭日志
    env.close()
    agent.save(CONFIG['MODEL_PATH'])
    writer.close()

if __name__ == "__main__":
    train()
