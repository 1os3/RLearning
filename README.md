# RL-Liquid Neural Network AutoDriving

## 技术框架

本项目基于液体神经网络（LNN）与强化学习，结合 AirSim 仿真环境，实现端到端的无人机视觉导航。核心技术栈如下：

- **仿真环境**：Unreal Engine + AirSim 插件，支持高保真物理与视觉。
- **神经网络**：液体神经网络（LNN），20神经元，原生PyTorch实现，适配Intel GPU（torch.xpu/oneAPI）。
- **强化学习**：自定义奖励结构，支持经验回放、epsilon贪婪探索、分项奖励统计。
- **数据流**：纯视觉输入+低维状态（速度/位移/姿态），模型输出三维动作。
- **日志监控**：TensorBoard全流程指标可视化。
- **高可维护性**：模块化目录结构，奖励、探索等参数集中于config.py，所有核心逻辑均有异常防御。

### 目录结构简要
```
RLearning/
├── lnn/                  # 液体神经网络核心模块
├── agent/                # 强化学习智能体
├── env/                  # Unreal/AirSim环境接口
├── train/                # 训练与评估脚本
├── utils/                # 工具包
├── requirements.txt      # 依赖
├── README.md
└── main.py               # 入口
```

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

（本说明已同步2025年4月最新项目结构与参数，若有环境或奖励机制调整请及时补充）