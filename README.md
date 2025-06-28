# 🧠 AI 模型实验仓库

这是一个多模型实验项目，包含多个深度学习与强化学习模型，涵盖图像识别、生成模型、自然语言翻译以及小游戏 AI 等方向。

---

## 📁 项目结构概览

models/ # 存储训练好的模型权重（如 model.pth）
pytorch/ # PyTorch 实现的各种模型和实验脚本
├── CGAN.py # 条件生成对抗网络
├── CNNet.py # 简单卷积网络
├── Decoder.py # 编码器-解码器结构
├── DiscoGAN.py # 图像到图像的风格迁移
├── NMT(resnet+CIFAR)-10.py # 基于 ResNet 的图像分类任务
├── Q-learning and SARSA.py # 强化学习算法实现
├── VAE.py # 变分自编码器
├── VGG.py # VGG 卷积神经网络
├── download.py # 图像下载或数据加载脚本
└── not done/ # 未完成的实验，如 DQN
x-game/ # 各类游戏 AI 实验（五子棋、贪吃蛇、小鸟等）
├── Gomoku.py # 五子棋基础版本
├── ai-bird-*.py # 基于 DQN 的小鸟游戏 AI
├── ai-snake.py # 贪吃蛇 AI 控制
├── two_player_pong.py# 双人乒乓游戏
report01, report2, report3 # 项目实验文档或总结
