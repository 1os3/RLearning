import os
import torch
import numpy as np
import random
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from config import CONFIG
from env.airsim_env import AirSimVisionEnv
from agent.vision_agent import VisionLNNAgent
from utils.vision_utils import preprocess_image, stack_state
from utils.amp_utils import autocast_xpu
import gc

# ===================== 1. 经验回放池 =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *transition):
        self.buffer.append(tuple(transition))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        fields = list(zip(*batch))
        result = []
        for i, field in enumerate(fields):
            # 第0项是stacked_state，需保持shape为(batch, C, H, W)
            if i == 0:
                arr = np.stack(field)
            else:
                arr = np.array(field)
                if arr.dtype == np.object_:
                    arr = np.asarray(list(arr), dtype=np.float32)
                else:
                    arr = arr.astype(np.float32)
            result.append(arr)
        return result
    def __len__(self):
        return len(self.buffer)

# ===================== 2. 奖励调度器 =====================
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
        if avg_approach < 0.5:
            self.config['REWARD_APPROACH_MAX'] = min(self.config['REWARD_APPROACH_MAX'] + 1, 20)
        if avg_reward < 0:
            self.config['PENALTY_STATIC'] = min(self.config['PENALTY_STATIC'] - 1, -20)
    def get(self, key):
        return self.config.get(key)

# ===================== 3. 奖励函数 =====================
def calc_reward(info, last_info=None, context=None, config=CONFIG, step_count=None):
    reward = config['REWARD_ALIVE']
    distance = float(info.get('distance', 1))
    # 距离奖励用平方递增，靠近目标更有动力
    reward += config['REWARD_DISTANCE'] * ((max(0, config['REWARD_DISTANCE_MAX'] - distance) / config['REWARD_DISTANCE_MAX']) ** 2)
    speed = float(info.get('speed', 0))
    if speed < config['REWARD_SPEED_TARGET']:
        reward -= abs(speed - config['REWARD_SPEED_TARGET']) * config['REWARD_SPEED_SCALE']
    z = float(info.get('z', config['REWARD_Z_TARGET']))
    reward -= abs(z - config['REWARD_Z_TARGET']) * config['REWARD_Z_SCALE']
    pitch = abs(float(info.get('pitch', 0)))
    roll = abs(float(info.get('roll', 0)))
    reward -= (pitch + roll) * config['REWARD_ATTITUDE_SCALE']
    if info.get('collision', False):
        reward += config['PENALTY_COLLISION']
    if info.get('offroad', False):
        reward += config['PENALTY_OFFROAD']
    # 静止与慢速惩罚强化
    if speed < 0.1:
        reward += config['PENALTY_SLOW']
    if abs(info.get('speed', 0)) < 0.05:
        reward += config.get('PENALTY_STATIC', -20.0)
    if context is not None and context.get('repeat_traj', False):
        reward += config.get('PENALTY_REPEAT_TRAJ', -10.0)
    if context is not None and context.get('action_switch', False):
        reward += config.get('PENALTY_ACTION_SWITCH', -1.0)
    max_speed = config['REWARD_SPEED_NORM']
    speed_reward = min(speed, max_speed) / max_speed * config.get('REWARD_SPEED_MAX', 2.0)
    reward += speed_reward
    # 靠近目标点方向速度奖励/惩罚
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
            approach_reward = min(v_proj, max_proj_speed) / max_proj_speed * config.get('REWARD_APPROACH_MAX', 10.0)
        else:
            approach_reward = v_proj * config.get('PENALTY_AWAY_SCALE', 10.0)
        reward += approach_reward
    return reward

# ===================== 4. 辅助判定 =====================
def check_static(speeds, config):
    return all(s < 0.05 for s in speeds[-config['STATIC_STEPS']:]) if len(speeds) >= config['STATIC_STEPS'] else False

def check_repeat_traj(positions, config):
    if len(positions) < config['REPEAT_TRAJ_WINDOW']:
        return False
    recent = positions[-config['REPEAT_TRAJ_WINDOW']:]
    arr = np.array(recent)
    dists = np.linalg.norm(arr - arr[-1], axis=1)
    return np.sum(dists < 0.5) > 3

def check_action_switch(actions, config):
    if len(actions) < config['ACTION_SWITCH_WINDOW']:
        return False
    recent = actions[-config['ACTION_SWITCH_WINDOW']:]
    return len(set(recent)) > 2

def check_collision_repeat(collisions, config):
    return sum(collisions[-config['COLLISION_REPEAT_WINDOW']:]) > 1

# ===================== 5. 训练状态保存/恢复 =====================
def save_train_state(replay, step_count, ep, epsilon, best_reward, path):
    torch.save({
        'replay': replay.buffer,
        'step_count': step_count,
        'episode': ep,
        'epsilon': epsilon,
        'best_reward': best_reward
    }, path)
    print(f'[Train] 已保存训练状态到 {path}')

def load_train_state(replay, path):
    if os.path.exists(path):
        state = torch.load(path)
        replay.buffer = state['replay']
        print(f'[Train] 已恢复训练状态: episode={state["episode"]}, step={state["step_count"]}, epsilon={state["epsilon"]}')
        return state['step_count'], state['episode'], state['epsilon'], state['best_reward']
    return 0, 0, CONFIG['EPSILON_START'], -float('inf')

# ===================== 6. 日志与TensorBoard =====================
def write_reward_items(writer, reward_items, step_count):
    for k, v in reward_items.items():
        writer.add_scalar(f'RewardItems/{k}', v, step_count)

def log_warnings(info, ep, step_count):
    if info.get('collision', False):
        print(f'[警告] 第{ep+1}集第{step_count}步发生碰撞')
    if info.get('offroad', False):
        print(f'[警告] 第{ep+1}集第{step_count}步偏离航线')

# ===================== 7. 状态采集与预处理 =====================
def collect_state(env, context, state_list):
    info = env.get_info(target_pos=context['target_pos']) if hasattr(env, 'get_info') else {}
    lowdim_keys = ['speed','x','y','z','pitch','roll','yaw']
    lowdim = np.array([float(info.get(k,0)) for k in lowdim_keys], dtype=np.float32)
    lowdim_tensor = torch.from_numpy(lowdim).unsqueeze(0)
    cur_pos = np.array([float(info.get('x', 0)), float(info.get('y', 0)), float(info.get('z', 0))], dtype=np.float32)
    delta_pos = cur_pos - np.array(context['init_pos'], dtype=np.float32)
    delta_pos_tensor = torch.from_numpy(delta_pos).unsqueeze(0)
    target_pos_tensor = torch.from_numpy(np.array(context['target_pos'], dtype=np.float32)).unsqueeze(0)
    elapsed_time = np.array([time.time() - context['start_time']], dtype=np.float32)
    elapsed_time_tensor = torch.from_numpy(elapsed_time).unsqueeze(0)
    distance = np.array([float(info.get('distance', 0))], dtype=np.float32)
    distance_tensor = torch.from_numpy(distance).unsqueeze(0)
    img = env.get_image()
    state = preprocess_image(img)
    state_list.pop(0)
    state_list.append(state)
    stacked_state = stack_state(state_list)
    return (stacked_state, lowdim_tensor, delta_pos_tensor, target_pos_tensor, elapsed_time_tensor, distance_tensor, info)

# ===================== 8. 模型推理与动作选择 =====================
def select_action(agent, stacked_state, epsilon, lowdim=None, delta_pos=None, target_pos=None, elapsed_time=None, distance=None):
    return agent.select_action(stacked_state, epsilon, lowdim=lowdim, delta_pos=delta_pos, target_pos=target_pos, elapsed_time=elapsed_time, distance=distance)

# ===================== 9. 训练主循环 =====================
# ===================== 学习率调度器 =====================
def get_scheduler(optimizer):
    sched_conf = CONFIG['LR_SCHEDULER']
    if sched_conf['type'] == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, factor=sched_conf['factor'], patience=sched_conf['patience'], min_lr=sched_conf['min_lr'], mode=sched_conf['mode'])
    elif sched_conf['type'] == 'StepLR':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=sched_conf.get('step_size', 20), gamma=sched_conf.get('factor', 0.5))
    elif sched_conf['type'] == 'CosineAnnealingLR':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=sched_conf.get('T_max', 100), eta_min=sched_conf.get('min_lr', 1e-6))
    else:
        return None

def train():
    seed = CONFIG.get('SEED', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = AirSimVisionEnv()
    device = torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    img = env.get_image()
    state = preprocess_image(img)
    state_list = [state for _ in range(CONFIG['STACK_SIZE'])]
    stacked_state = stack_state(state_list)
    sample_info = env.get_info() if hasattr(env, 'get_info') else {}
    lowdim_keys = ['speed','x','y','z','pitch','roll','yaw']
    lowdim_dim = len([k for k in lowdim_keys if k in sample_info]) if sample_info else 9
    agent = VisionLNNAgent(input_shape=stacked_state.shape, action_dim=3, lowdim_dim=lowdim_dim, device=device)
    replay = ReplayBuffer(CONFIG['REPLAY_BUFFER_SIZE'])
    epsilon = CONFIG['EPSILON_START']
    best_reward = -float('inf')
    start_ep = 0
    train_state_path = CONFIG.get('TRAIN_STATE_PATH', 'train_state.pth')
    if os.path.exists(train_state_path):
        step_count, start_ep, epsilon, best_reward = load_train_state(replay, train_state_path)
    elif os.path.exists(CONFIG['MODEL_PATH']):
        agent.load(CONFIG['MODEL_PATH'])
        print(f"[Train] 加载断点模型: {CONFIG['MODEL_PATH']}")
    os.makedirs(os.path.dirname(CONFIG['MODEL_PATH']), exist_ok=True)
    scheduler = None
    if hasattr(agent, 'optimizer'):
        scheduler = get_scheduler(agent.optimizer)
    writer = SummaryWriter(log_dir="runs/lnn_rl")
    reward_scheduler = RewardScheduler(CONFIG.copy())
    train_step = 0  # 训练全局步数
    exploration_steps = CONFIG.get('EXPLORATION_STEPS', 20000)
    for ep in trange(start_ep, CONFIG['MAX_EPISODES'], desc="Episode"):
        try:
            env.reset()
            agent.reset_state()
            state_list = [state for _ in range(CONFIG['STACK_SIZE'])]
            stacked_state = stack_state(state_list)
            total_reward = 0
            done = False
            last_info = None
            losses = []
            drone_init_info = env.get_info()
            drone_init_pos = [float(drone_init_info['x']), float(drone_init_info['y']), float(drone_init_info['z'])]
            target_pos = [drone_init_pos[0] + 200.0, drone_init_pos[1], drone_init_pos[2] + 2.0]
            context = {
                'target_pos': target_pos,
                'init_pos': drone_init_pos,
                'start_time': time.time(),
                'static_steps': 0,
                'repeat_traj': False,
                'action_switch': False,
                'total_distance': 0.0,
                'collision_count': 0,
                # 新增静止检测字段
                'no_move_start_time': None,
                'no_move_distance': 0.0
            }
            speeds, positions, actions, collisions = [], [], [], []
            min_dist = float('inf')
            no_progress_steps = 0
            for step_count in range(CONFIG['MAX_STEPS']):
                prev_stacked_state = stacked_state.copy() if hasattr(stacked_state, 'copy') else np.copy(stacked_state)
                stacked_state, lowdim_tensor, delta_pos_tensor, target_pos_tensor, elapsed_time_tensor, distance_tensor, info = collect_state(env, context, state_list)
                if isinstance(prev_stacked_state, np.ndarray):
                    prev_stacked_state_tensor = torch.from_numpy(prev_stacked_state).to(device=device, dtype=torch.float32)
                else:
                    prev_stacked_state_tensor = prev_stacked_state.to(device=device, dtype=torch.float32)
                # 打印每步传递给模型的全部非图像输入（低维、位移、目标点、时间、距离、隐状态等）
                print(f"[Step {step_count}] lowdim: {lowdim_tensor.detach().cpu().numpy().round(3)}")
                print(f"    delta_pos: {delta_pos_tensor.detach().cpu().numpy().round(3)}")
                print(f"    target_pos: {target_pos_tensor.detach().cpu().numpy().round(3)}")
                print(f"    elapsed_time: {elapsed_time_tensor.detach().cpu().numpy().round(3)}")
                print(f"    distance: {distance_tensor.detach().cpu().numpy().round(3)}")
                # 打印隐状态（如有）
                if hasattr(agent, 'lnn') and hasattr(agent.lnn, 'h'):
                    h_val = agent.lnn.h.detach().cpu().numpy() if agent.lnn.h is not None else None
                    print(f"    lnn_hidden_state: {h_val}")
                print(f"    info: speed={info.get('speed'):.3f}, pos=({info.get('x'):.2f},{info.get('y'):.2f},{info.get('z'):.2f}), target={context['target_pos']}")
                # 动态调整epsilon，前exploration_steps步全探索
                if train_step < exploration_steps:
                    epsilon = 1.0
                else:
                    epsilon = max(CONFIG['EPSILON_END'], CONFIG['EPSILON_START'] - (train_step - exploration_steps) / CONFIG['EPSILON_DECAY'])
                action = select_action(agent, prev_stacked_state_tensor, epsilon, lowdim=lowdim_tensor, delta_pos=delta_pos_tensor, target_pos=target_pos_tensor, elapsed_time=elapsed_time_tensor, distance=distance_tensor)
                vx, vy, vz = [float(x) for x in action] if len(action) == 3 else (0.0, 0.0, 0.0)
                env.step(vx, vy, vz)
                info = env.get_info()
                speeds.append(float(info['speed']))
                positions.append([float(info['x']), float(info['y']), float(info['z'])])
                actions.append(tuple(action))
                collisions.append(int(info['collision']))
                context['total_distance'] += float(info['speed'])
                context['static_steps'] = context['static_steps']+1 if float(info['speed']) < 0.05 else 0
                context['repeat_traj'] = check_repeat_traj(positions, CONFIG)
                context['action_switch'] = check_action_switch(actions, CONFIG)
                context['collision_count'] = sum(collisions[-CONFIG['COLLISION_REPEAT_WINDOW']:])
                # 静止检测逻辑
                if float(info['speed']) < 0.05:
                    if context['no_move_start_time'] is None:
                        context['no_move_start_time'] = time.time()
                        context['no_move_distance'] = 0.0
                    else:
                        # 计算相邻两步距离
                        if len(positions) > 1:
                            d = np.linalg.norm(np.array(positions[-1]) - np.array(positions[-2]))
                            context['no_move_distance'] += d
                else:
                    context['no_move_start_time'] = None
                    context['no_move_distance'] = 0.0
                # 距离进步检测
                cur_dist = float(info.get('distance', 1e6))
                if cur_dist < min_dist - 0.1:
                    min_dist = cur_dist
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1
                # 终止条件1：500步距离无进步
                if no_progress_steps > 500:
                    print(f'[Train] 连续500步距离未减小，提前终止本集')
                    break
                # 终止条件2：静止超时且累计移动距离过小
                if (context['no_move_start_time'] is not None and
                    (time.time() - context['no_move_start_time'] > CONFIG.get('NO_MOVE_WINDOW', 300)) and
                    (context['no_move_distance'] < CONFIG.get('NO_MOVE_DIST', 10.0))):
                    print(f'[Train] 超过{CONFIG.get("NO_MOVE_WINDOW", 300)}秒累计未移动{CONFIG.get("NO_MOVE_DIST", 10.0)}米，提前终止本集')
                    break
                # 终止条件3：loss爆炸
                if len(losses) > 0 and (np.isnan(losses[-1]) or losses[-1] > 1e4):
                    print(f'[Train] Loss爆炸，提前终止本集')
                    break
                if float(info['distance']) < 1.0:
                    total_reward += 20.0
                    done = True
                    # 到达目标点，自动保存模型和训练状态并优雅退出
                    print(f'[Train] 已到达目标点，保存模型与训练状态并退出')
                    agent.save(CONFIG['MODEL_PATH'])
                    save_train_state(replay, train_step, ep, epsilon, best_reward, CONFIG.get('TRAIN_STATE_PATH', 'train_state.pth'))
                    import sys
                    sys.exit(0)
                else:
                    done = False
                reward = calc_reward(info, last_info, context, config=reward_scheduler.config, step_count=step_count)
                total_reward += reward
                last_info = info.copy()
                # 经验池推送
                replay.push(prev_stacked_state, action, reward, stacked_state, delta_pos_tensor, target_pos_tensor, elapsed_time_tensor, lowdim_tensor, done)
                # 视觉信息写入TensorBoard（每隔20步）
                if train_step % 20 == 0 and isinstance(stacked_state, torch.Tensor):
                    img_np = stacked_state[0].detach().cpu().numpy() if stacked_state.dim() == 4 else stacked_state.detach().cpu().numpy()
                    img_np = img_np.astype(np.float32)
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    # 兼容多帧堆叠与单帧RGB输入
                    if img_np.ndim == 3:
                        # 如果是多帧堆叠，取前三通道（RGB）
                        if img_np.shape[0] > 3:
                            img_np = img_np[:3, :, :]
                        # 如果是单通道，重复三次
                        elif img_np.shape[0] == 1:
                            img_np = np.repeat(img_np, 3, axis=0)
                    writer.add_image('Vision/state', img_np, train_step, dataformats='CHW')
                # reward/loss写入TensorBoard（step全局唯一）
                global_step = ep * CONFIG['MAX_STEPS'] + step_count
                writer.add_scalar('Reward/step', reward, global_step)
                writer.add_scalar('Reward/episode_cum', total_reward, global_step)
                writer.add_scalar('Distance/step', info.get('distance', 0), global_step)
                writer.add_scalar('Speed/step', info.get('speed', 0), global_step)
                writer.add_scalar('Epsilon', epsilon, global_step)
                reward_items = {
                    'reward': reward,
                    'distance': info.get('distance', 0),
                    'speed': info.get('speed', 0),
                    'static_steps': context['static_steps'],
                    'total_reward': total_reward
                }
                write_reward_items(writer, reward_items, global_step)
                log_warnings(info, ep, step_count)
                print(f"[Ep {ep+1} | Step {step_count+1}] Reward: {reward:.2f}, TotalReward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Speed: {info.get('speed', 0):.2f}, Distance: {info.get('distance', 0):.2f}, Collision: {info.get('collision', False)}, Offroad: {info.get('offroad', False)}")
                # 多步经验回放采样
                train_times = 2
                for _ in range(train_times):
                    if len(replay) > CONFIG['BATCH_SIZE']:
                        batch = replay.sample(CONFIG['BATCH_SIZE'])
                        b_s, b_a, b_r, b_ns, b_delta_pos, b_target_pos, b_elapsed_time, b_lowdim, b_d = batch
                        b_s = torch.from_numpy(b_s).to(device=device, dtype=torch.float32)
                        b_a = torch.from_numpy(b_a).to(device=device, dtype=torch.float32)
                        b_r = torch.from_numpy(b_r).to(device=device, dtype=torch.float32)
                        b_ns = torch.from_numpy(b_ns).to(device=device, dtype=torch.float32) if b_ns is not None else None
                        b_delta_pos = torch.from_numpy(b_delta_pos).to(device=device, dtype=torch.float32)
                        b_target_pos = torch.from_numpy(b_target_pos).to(device=device, dtype=torch.float32)
                        b_elapsed_time = torch.from_numpy(b_elapsed_time).to(device=device, dtype=torch.float32)
                        b_lowdim = torch.from_numpy(b_lowdim).to(device=device, dtype=torch.float32)
                        b_d = torch.from_numpy(b_d).to(device=device, dtype=torch.float32)
                        with autocast_xpu():
                            with torch.no_grad():
                                out, _ = agent.forward(b_ns, lowdim=b_lowdim, delta_pos=b_delta_pos, target_pos=b_target_pos, elapsed_time=b_elapsed_time)
                                q_next = out["q"] if isinstance(out, dict) and "q" in out else out
                                q_target = b_r + CONFIG['GAMMA'] * (1 - b_d) * q_next.squeeze()
                            out_pred, _ = agent.forward(b_s, lowdim=b_lowdim, delta_pos=b_delta_pos, target_pos=b_target_pos, elapsed_time=b_elapsed_time)
                            q_pred = out_pred["q"] if isinstance(out_pred, dict) and "q" in out_pred else out_pred
                            loss = torch.nn.functional.mse_loss(q_pred.squeeze(), q_target)
                            if torch.isnan(loss) or loss.item() > 1e4:
                                print(f'[Train] Loss异常，跳过本步')
                                continue
                            agent.optimizer.zero_grad()
                            loss.backward()
                            agent.optimizer.step()
                            losses.append(loss.item())
                            train_step += 1
                            if train_step % 20 == 0:
                                writer.add_scalar('Loss/train', loss.item(), train_step)
                                writer.add_scalar('Q/mean', q_pred.mean().item(), train_step)
                if hasattr(agent, 'update_epsilon'):
                    epsilon = agent.update_epsilon(epsilon)
                else:
                    eps_min = CONFIG.get('EPSILON_MIN', CONFIG.get('EPSILON_END', 0.01))
                    epsilon = max(eps_min, epsilon * CONFIG['EPSILON_DECAY'])
                if done:
                    break
                # 自动保存模型，每200步
                if (step_count + 1) % 200 == 0:
                    save_train_state(replay, step_count + 1, ep, epsilon, best_reward, CONFIG['TRAIN_STATE_PATH'])
            # episode级别日志
            if len(losses) > 0:
                writer.add_scalar('Loss/episode_avg', np.mean(losses), ep)
            writer.add_scalar('Reward/episode', total_reward, ep)
            writer.add_scalar('Distance/episode_min', min_dist, ep)
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(CONFIG['MODEL_PATH'])
            if (ep + 1) % 200 == 0:
                # 保存模型和训练状态快照
                agent.save(CONFIG['MODEL_PATH'].replace('.pth', f'_ep{ep+1}.pth'))
                save_train_state(replay, step_count, ep, epsilon, best_reward, train_state_path.replace('.pth', f'_ep{ep+1}.pth'))
                # 额外复制一份模型到独立目录（如snapshots/）
                import shutil
                snap_dir = os.path.join(os.path.dirname(CONFIG['MODEL_PATH']), 'snapshots')
                os.makedirs(snap_dir, exist_ok=True)
                snap_path = os.path.join(snap_dir, f'model_ep{ep+1}.pth')
                shutil.copy(CONFIG['MODEL_PATH'], snap_path)
            save_train_state(replay, step_count, ep, epsilon, best_reward, train_state_path)
            gc.collect()
            if hasattr(torch, 'xpu'):
                torch.xpu.empty_cache()
        except Exception as e:
            print(f'[Train] Episode {ep} 异常: {e}')
            continue
    writer.close()

if __name__ == "__main__":
    train()
