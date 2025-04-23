# RL-Liquid Neural Network AutoDriving

## 技术框架

本项目基于液体神经网络（LNN）与强化学习，结合 AirSim 仿真环境，实现端到端的无人机视觉导航。核心技术栈如下：

- **仿真环境**：Unreal Engine + AirSim 插件，支持高保真物理与视觉。
- **神经网络**：液体神经网络（LNN），20神经元，原生PyTorch实现，适配Intel GPU（torch.xpu/oneAPI）。
- **强化学习**：自定义奖励结构，支持经验回放、epsilon贪婪探索、分项奖励统计。
- **数据流**：纯视觉输入+低维状态（速度/位移/姿态），模型输出三维动作。
- **日志监控**：TensorBoard全流程指标可视化。
- **高可维护性**：模块化目录结构，奖励、探索等参数集中于config.py，所有核心逻辑均有异常防御。

### 目录结构（2025最新版）
```
RLearning/
├── agent/                  # 智能体模块（vision_agent.py）
│   └── vision_agent.py
├── env/                    # AirSim环境接口（airsim_env.py）
│   └── airsim_env.py
├── lnn/                    # 液体神经网络实现
│   ├── lnn_cell.py
│   └── lnn_network.py
├── train/                  # 训练主循环
│   └── train.py
├── utils/                  # 工具包（奖励调优、图像处理等）
│   ├── amp_utils.py
│   ├── auto_reward_tuner.py
│   └── vision_utils.py
├── config.py               # 全局超参数与奖励配置
├── main.py                 # 入口脚本（支持命令行参数）
├── requirements.txt        # Python依赖
├── run.bat                 # Windows一键训练脚本
├── run_tensorboard.bat     # 一键启动TensorBoard
├── README.md               # 项目说明文档
└── .gitignore
```
（注：__pycache__等缓存目录已省略）

## 技术框架

本项目采用模块化、可维护的架构，核心技术与分层如下：

### 1. 仿真与环境层

- **AirSim + Unreal Engine**：提供高保真物理、视觉仿真，支持多无人机、多场景、可定制传感器。通过 TCP/REST API 与 Python 环境交互。
- **env/**：对 AirSim 原生 API 进行高层封装，提供 reset、step、get_obs、采样目标点等通用接口，兼容多种仿真设置。

### 2. 智能体与算法层

- **agent/**：强化学习智能体实现，核心为 VisionLNNAgent，支持：
  - 视觉输入（RGB 图像，归一化处理）
  - 低维输入（速度、位置、姿态、距离等）
  - 动作输出（速度或加速度指令）
  - epsilon-greedy 策略、经验回放、损失计算等
- **lnn/**：液体神经网络（LNN）实现，具备递归状态、时间动态建模能力，适配 PyTorch/Intel oneAPI。
- **utils/**：包含经验回放缓存、图像处理、日志、TensorBoard 记录等通用工具。

### 3. 训练与评估层

- **train/**：主训练循环，负责：
  - 环境与智能体初始化
  - 采样与奖励计算（含距离、速度、静止、碰撞等多维奖励/惩罚）
  - 探索-利用切换（支持 EXPLORATION_STEPS 配置）
  - 终止条件（如静止超时、loss 爆炸、到达目标点等）
  - 自动保存模型与训练状态，断点续训
  - 日志输出与异常防御

### 4. 配置与可维护性

- **config.py**：集中管理所有超参数、奖励结构、探索步数等，便于实验调优与复现实验。
- **异常防御**：所有核心流程均有 try/except，训练异常时自动保存模型和状态，便于恢复。
- **高可扩展性**：各层接口清晰，便于扩展新奖励、新输入特征或替换 RL 算法。

---

**整体流程**：  
用户启动训练脚本 → 连接 AirSim 仿真 → 智能体采集视觉/状态 → 计算动作 → 发送动作控制 → 环境反馈奖励/新状态 → 训练更新 → 日志与模型保存。

---

如需进一步扩展（如多智能体、分布式训练、复杂奖励结构等），仅需在相应模块增补接口即可，无需重构主流程。  
如遇具体技术难题，建议先查阅本项目各子模块 README 或 AirSim 官方文档。

---

## 部署方式

### 1. 环境准备
- 推荐 Python 3.8+，建议使用虚拟环境（venv/anaconda）。
- 安装依赖：
```bash
pip install -r requirements.txt
```
- 配置好 AirSim（Unreal Engine 插件），并启动仿真场景。

### 2. 训练与推理
- 直接运行：
```bash
python train/train.py
```
- 支持断点续训。
- 训练日志与奖励分项自动写入 TensorBoard，运行 `tensorboard --logdir runs/` 查看。

### 3. Intel GPU 支持
- PyTorch >=2.0 已原生支持 Intel GPU（oneAPI），代码自动检测 xpu 设备。
- 若无 Intel GPU，将自动回退到 CPU。

### 4. 推荐配置
- 训练参数、奖励结构等全部集中于 config.py，便于实验与调优。
- 经验回放缓存如遇类型异常建议清空，避免历史数据污染。
- 代码已加入类型防御与异常捕获，便于定位调试。
- 如需自定义奖励结构，建议在 `train/train.py:calc_reward` 处修改。

### 5. 其它
- 训练/推理均支持纯视觉端到端与低维状态联合输入。
- 所有模块均可独立测试。

---

## AirSim 仿真环境详细说明

### 1. AirSim 介绍
[AirSim](https://github.com/microsoft/AirSim) 是微软开源的高保真无人机/自动驾驶仿真平台，基于 Unreal Engine，支持真实物理、传感器、视觉和多种车辆类型。

- **本项目用途**：用于训练无人机在复杂环境下的视觉导航能力。
- **支持场景**：推荐使用 AirSim 自带的 Blocks/AbandonedPark 等地图，或自定义 Unreal 地图。

### 2. AirSim 安装与配置
1. 安装 Unreal Engine（推荐 4.27 版）。
2. 下载并编译 AirSim 插件（详见 AirSim 官方文档）。
3. 将 AirSim 插件拷贝到 Unreal 项目 `Plugins` 目录，打开地图并运行。
4. 配置 `settings.json`，如：
   ```json
   {
     "SettingsVersion": 1,
     "SimMode": "Multirotor",
     "Vehicles": {
       "Drone1": {
         "VehicleType": "SimpleFlight",
         "AutoCreate": true,
         "Cameras": {
           "front_center": {
             "CaptureSettings": [
               { "ImageType": 0, "Width": 84, "Height": 84, "FOV_Degrees": 90 }
             ]
           }
         }
       }
     },
     "ClockSpeed": 1.0
   }
   ```
5. 启动 Unreal 地图，确保 AirSim 控制台无报错。

### 3. RL 环境接口
- 本项目通过 `env/airsim_env.py` 封装 AirSim API，支持：
  - 图像采集（84x84 RGB）
  - 低维状态（速度、位置、姿态）
  - 动作控制（速度/加速度指令）
  - 目标点采样与重置
- 可根据 `settings.json` 调整相机参数、仿真速度等。

### 4. 常见问题
- **无法连接 AirSim**：检查 Unreal 场景已运行，端口未被占用。
- **图像尺寸不符**：确保 `settings.json` 配置与 `config.py` 保持一致。
- **仿真掉帧/卡顿**：建议关闭 Unreal Editor 的实时渲染或降低分辨率。
- **多机仿真**：可在 `settings.json` 配置多个 Drone。

---

## 技术框架（补充细节）

- **agent/**：强化学习智能体，核心为 VisionLNNAgent，支持视觉+低维联合输入。
- **lnn/**：液体神经网络实现，支持递归状态与时间建模。
- **train/**：主训练循环，含静止检测、奖励调度、断点续训、自动保存等。
- **env/**：对 AirSim API 的高层封装。
- **utils/**：包括图像预处理、经验回放、日志等工具。
- **config.py**：所有超参数、奖励项集中配置，便于实验调优。
- **异常防御**：所有核心逻辑均有 try/except，训练中断自动保存。

---

如需自定义仿真环境、扩展传感器或更换地图，只需修改 AirSim 配置和 env/ 相关接口。
如遇部署或环境问题，建议优先查阅 AirSim 官方文档和本项目 issues。

---

## 代码库上传与同步

### 上传到 GitHub
1. **初始化 git 仓库**（如未初始化）：
   ```bash
   git init
   ```
2. **添加远程仓库**：
   ```bash
   git remote add origin https://github.com/1os3/RLearning.git
   ```
3. **添加所有文件（排除 Airsim 与 venv 文件夹）**：
   ```bash
   git add .
   git reset Airsim
   git reset venv
   ```
   或编辑 `.gitignore` 文件，加入：
   ```
   Airsim/
   venv/
   ```
4. **提交并推送**：
   ```bash
   git commit -m "init: RL-Liquid Neural Network AutoDriving"
   git push -u origin master
   ```

---

如遇依赖、环境或部署问题，请查阅 requirements.txt 与 config.py，或在 Issues 区留言。