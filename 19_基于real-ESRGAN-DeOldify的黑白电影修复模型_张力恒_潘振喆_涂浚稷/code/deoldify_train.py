import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm

# 创建必要的目录结构
os.makedirs('training_data/video_frames', exist_ok=True)
os.makedirs('training_data/processed_frames', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)


class VideoFrameDataset(Dataset):
    def __init__(self, video_path, frame_interval=10, max_frames=1000):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.frames = self.extract_frames()

    def extract_frames(self):
        """从视频中提取帧"""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_interval == 0:
                # 转换为RGB并调整大小
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (256, 256))
                frames.append(frame_resized)

            frame_count += 1
            if len(frames) >= self.max_frames:
                break

        cap.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx].astype(np.float32) / 255.0

        # 转换为LAB颜色空间
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]  # 亮度通道作为输入
        ab_channels = lab[:, :, 1:]  # 颜色通道作为目标

        # 转换为tensor
        l_tensor = torch.FloatTensor(l_channel).unsqueeze(0)  # [1, H, W]
        ab_tensor = torch.FloatTensor(ab_channels).permute(2, 0, 1)  # [2, H, W]

        return l_tensor, ab_tensor


class DataAugmentation:
    """数据增强类"""

    @staticmethod
    def random_crop(image, size=(224, 224)):
        h, w = image.shape[:2]
        x = np.random.randint(0, w - size[1])
        y = np.random.randint(0, h - size[0])
        return image[y:y + size[0], x:x + size[1]]

    @staticmethod
    def random_flip(image):
        if np.random.random() > 0.5:
            return cv2.flip(image, 1)
        return image

    @staticmethod
    def adjust_brightness(image, factor=0.2):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * (1 + factor * (np.random.random() - 0.5))
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


#这块是模型架构
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DeOldifyGenerator(nn.Module):
    def __init__(self):
        super(DeOldifyGenerator, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(8)]
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 2, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch_size, 1, H, W] 灰度图像
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x  # [batch_size, 2, H, W] ab通道

#这部分是训练器
class DeOldifyTrainer:
    def __init__(self, pretrained_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化模型
        self.generator = DeOldifyGenerator().to(self.device)

        # 加载预训练模型
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model from {pretrained_model_path}")
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print("Pretrained model loaded successfully")

        # 损失函数和优化器
        self.criterion = nn.L1Loss()  # 使用L1损失保持颜色稳定性
        self.optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # 训练历史
        self.train_losses = []

    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for i, (l_channel, ab_channels) in enumerate(pbar):
            l_channel = l_channel.to(self.device)
            ab_channels = ab_channels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            predicted_ab = self.generator(l_channel)

            # 计算损失
            loss = self.criterion(predicted_ab, ab_channels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        epoch_loss = running_loss / len(dataloader)
        self.train_losses.append(epoch_loss)

        return epoch_loss

    def validate(self, dataloader):
        self.generator.eval()
        val_loss = 0.0

        with torch.no_grad():
            for l_channel, ab_channels in dataloader:
                l_channel = l_channel.to(self.device)
                ab_channels = ab_channels.to(self.device)

                predicted_ab = self.generator(l_channel)
                loss = self.criterion(predicted_ab, ab_channels)
                val_loss += loss.item()

        return val_loss / len(dataloader)

    def save_checkpoint(self, epoch, loss, checkpoint_dir='checkpoints'):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses
        }

        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self, train_loader, val_loader, epochs=50, save_interval=5):
        print("Starting training...")

        for epoch in range(1, epochs + 1):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step()

            print(
                f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.8f}')

            # 保存checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, val_loss)


def main():
    # 配置参数
    config = {
        'video_path': 'path/to/your/training_video.mp4',  # 替换为您的训练视频路径
        'pretrained_model_path': 'models/ColorizeVideo_gen.pth',  # 您的预训练模型路径
        'batch_size': 8,
        'epochs': 50,
        'frame_interval': 10,
        'max_frames': 1000
    }

    # 创建数据集
    print("Creating dataset...")
    dataset = VideoFrameDataset(
        video_path=config['video_path'],
        frame_interval=config['frame_interval'],
        max_frames=config['max_frames']
    )

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 初始化训练器
    trainer = DeOldifyTrainer(pretrained_model_path=config['pretrained_model_path'])

    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        save_interval=5
    )

    print("Training completed!")


if __name__ == "__main__":
    main()