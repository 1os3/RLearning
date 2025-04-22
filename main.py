import sys
import torch
import argparse
from config import CONFIG

def check_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("[INFO] Intel GPU (xpu) 可用，当前设备:", torch.xpu.current_device())
    else:
        print("[INFO] Intel GPU (xpu) 不可用，使用CPU。")

def infer():
    from env.airsim_env import AirSimVisionEnv
    from agent.vision_agent import VisionLNNAgent
    from utils.vision_utils import preprocess_image, stack_state
    import time
    import os
    env = AirSimVisionEnv()
    agent = VisionLNNAgent(input_shape=(CONFIG['STACK_SIZE'], 84, 84), action_dim=3)
    if os.path.exists(CONFIG['MODEL_PATH']):
        agent.load(CONFIG['MODEL_PATH'])
        print(f"[Infer] 加载模型: {CONFIG['MODEL_PATH']}")
    else:
        print(f"[Infer] 未找到模型: {CONFIG['MODEL_PATH']}")
        return
    state_list = []
    env.reset()
    img = env.get_image()
    state = preprocess_image(img)
    for _ in range(CONFIG['STACK_SIZE']):
        state_list.append(state)
    stacked_state = stack_state(state_list)
    agent.eval()
    total_reward = 0
    step = 0
    done = False
    while not done:
        action = agent.select_action(stacked_state, epsilon=0.0)
        throttle = 0.5
        steering = (action - 1) * 0.5
        env.step(throttle, steering)
        next_img = env.get_image()
        next_state = preprocess_image(next_img)
        state_list.pop(0)
        state_list.append(next_state)
        stacked_state = stack_state(state_list)
        # info可扩展，done由用户人工输入
        print(f"[Infer] Step {step+1}: Action={action}, Throttle={throttle}, Steering={steering}")
        user_input = input("是否终止本回合？(y/n): ").strip().lower()
        if user_input == 'y':
            done = True
        step += 1
        time.sleep(0.05)
    env.close()
    print("推理结束。")

def export_model():
    from agent.vision_agent import VisionLNNAgent
    from config import CONFIG
    import torch
    agent = VisionLNNAgent(input_shape=(CONFIG['STACK_SIZE'], 84, 84), action_dim=3, lowdim_dim=9)
    agent.load(CONFIG['MODEL_PATH'])
    agent.export_onnx('lnn_agent.onnx', input_shape=(1, CONFIG['STACK_SIZE'], 84, 84), lowdim_shape=(1, 9))

def main():
    parser = argparse.ArgumentParser(description="LNN视觉自动驾驶强化学习")
    parser.add_argument('--mode', type=str, default='infer', choices=['infer', 'export', 'train'], help='运行模式')
    args = parser.parse_args()
    check_device()
    if args.mode == 'infer':
        infer()
    elif args.mode == 'export':
        export_model()
    elif args.mode == 'train':
        from train.train import train
        train()
    else:
        print('Unknown mode')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Main] 程序异常: {e}")
