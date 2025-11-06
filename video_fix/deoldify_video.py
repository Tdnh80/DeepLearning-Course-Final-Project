# complete_video_colorizer.py
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from tqdm import tqdm
import subprocess
import argparse


class DeOldifyModel(nn.Module):
    """DeOldify 模型定义"""

    def __init__(self):
        super(DeOldifyModel, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 下采样
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 下采样
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 残差块
        self.residual_blocks = self._make_residual_blocks(256, 8)

        # 解码器部分
        self.decoder = nn.Sequential(
            # 上采样
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 上采样
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 输出层
            nn.Conv2d(64, 2, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def _make_residual_blocks(self, channels, num_blocks):
        """创建残差块序列"""
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        """前向传播"""
        # x: [batch_size, 1, height, width] 灰度图像
        x = self.encoder(x)  # 编码
        x = self.residual_blocks(x)  # 残差连接
        x = self.decoder(x)  # 解码
        return x  # 输出: [batch_size, 2, height, width] ab通道


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out


class VideoColorizer:
    """视频上色器主类"""

    def __init__(self, model_path, device='auto'):
        """
        初始化上色器

        Args:
            model_path: 模型文件路径
            device: 运行设备 ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        print(f"✅ 上色器初始化完成，使用设备: {self.device}")

    def _setup_device(self, device):
        """设置运行设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)

    def _load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")

        # 创建模型实例
        model = DeOldifyModel()

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'generator_state_dict' in checkpoint:
            # 训练保存的checkpoint
            model.load_state_dict(checkpoint['generator_state_dict'])
        elif 'state_dict' in checkpoint:
            # 其他格式的checkpoint
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 直接是模型权重
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()  # 设置为评估模式

        print(f"✅ 模型加载成功: {model_path}")
        return model

    def preprocess_frame(self, frame):
        """
        预处理视频帧

        Args:
            frame: 输入帧 (BGR格式)

        Returns:
            input_tensor: 模型输入张量
            original_size: 原始尺寸
            gray_frame: 灰度帧
        """
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame

        # 保存原始尺寸
        original_size = gray_frame.shape

        # 调整大小为模型输入尺寸
        gray_resized = cv2.resize(gray_frame, (256, 256))

        # 归一化到 [0, 1]
        gray_normalized = gray_resized.astype(np.float32) / 255.0

        # 转换为PyTorch张量: [1, 1, 256, 256]
        input_tensor = torch.FloatTensor(gray_normalized).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        return input_tensor, original_size, gray_resized

    def postprocess_frame(self, l_channel, ab_pred, original_size):
        """
        后处理上色结果

        Args:
            l_channel: L通道 (亮度)
            ab_pred: 预测的ab通道
            original_size: 原始图像尺寸

        Returns:
            colorized_frame: 上色后的BGR图像
        """
        # 将预测结果转换为numpy数组
        ab_pred_np = ab_pred.cpu().numpy()[0].transpose(1, 2, 0)  # [2, H, W] -> [H, W, 2]

        # 反归一化ab通道: 从 [-1, 1] 到 [0, 255]
        ab_uint8 = ((ab_pred_np + 1) * 127.5).astype(np.uint8)

        # 准备L通道
        l_uint8 = (l_channel * 255).astype(np.uint8)

        # 合并LAB通道
        lab_image = np.zeros((256, 256, 3), dtype=np.uint8)
        lab_image[:, :, 0] = l_uint8  # L通道
        lab_image[:, :, 1:] = ab_uint8  # AB通道

        # 转换回BGR颜色空间
        bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        # 调整回原始尺寸
        if original_size != (256, 256):
            bgr_image = cv2.resize(bgr_image, (original_size[1], original_size[0]))

        return bgr_image

    def colorize_frame(self, frame):
        """
        对单帧进行上色

        Args:
            frame: 输入帧 (BGR格式)

        Returns:
            colorized_frame: 上色后的帧
        """
        with torch.no_grad():  # 禁用梯度计算
            # 预处理
            input_tensor, original_size, gray_resized = self.preprocess_frame(frame)

            # 模型预测
            ab_pred = self.model(input_tensor)

            # 后处理
            colorized_frame = self.postprocess_frame(gray_resized, ab_pred, original_size)

            return colorized_frame

    def process_video(self, input_path, output_path, frame_interval=1, max_frames=None, output_fps=None):
        """
        处理整个视频

        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            frame_interval: 帧间隔 (1=处理每一帧)
            max_frames: 最大处理帧数
            output_fps: 输出视频帧率
        """
        # 检查输入文件
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"❌ 输入视频不存在: {input_path}")

        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"❌ 无法打开视频文件: {input_path}")

        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置输出帧率
        if output_fps is None:
            output_fps = original_fps / frame_interval

        print("=" * 50)
        print("📹 视频信息:")
        print(f"   输入文件: {input_path}")
        print(f"   分辨率: {width} x {height}")
        print(f"   原始帧率: {original_fps:.2f} FPS")
        print(f"   总帧数: {total_frames}")
        print(f"   帧间隔: {frame_interval}")
        print(f"   输出帧率: {output_fps:.2f} FPS")
        print("=" * 50)

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        if not out.isOpened():
            raise ValueError(f"❌ 无法创建输出视频文件: {output_path}")

        frame_count = 0
        processed_count = 0

        print("🎨 开始视频上色处理...")

        # 创建进度条
        with tqdm(total=total_frames, desc="处理进度") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 按间隔处理帧
                if frame_count % frame_interval == 0:
                    try:
                        # 上色处理
                        colorized_frame = self.colorize_frame(frame)

                        # 写入输出视频
                        out.write(colorized_frame)

                        processed_count += 1

                        # 达到最大帧数限制时停止
                        if max_frames and processed_count >= max_frames:
                            print(f"⚠️ 达到最大帧数限制: {max_frames}")
                            break

                    except Exception as e:
                        print(f"❌ 处理第 {frame_count} 帧时出错: {e}")
                        # 出错时写入原始帧
                        out.write(frame)

                frame_count += 1
                pbar.update(1)

        # 释放资源
        cap.release()
        out.release()

        print("=" * 50)
        print("✅ 处理完成!")
        print(f"   处理帧数: {processed_count}/{total_frames}")
        print(f"   输出文件: {output_path}")
        print("=" * 50)

    def process_image(self, input_path, output_path):
        """
        处理单张图片

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"❌ 输入图片不存在: {input_path}")

        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"❌ 无法读取图片: {input_path}")

        print(f"🖼️  处理图片: {input_path}")

        # 上色处理
        colorized_image = self.colorize_frame(image)

        # 保存结果
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, colorized_image)
        print(f"✅ 上色图片已保存: {output_path}")

        return colorized_image


def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='DeOldify 视频上色工具')
    parser.add_argument('--model', '-m', required=True, help='模型文件路径')
    parser.add_argument('--input', '-i', required=True, help='输入视频或图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出路径')
    parser.add_argument('--frame_interval', '-f', type=int, default=1, help='帧间隔 (默认: 1)')
    parser.add_argument('--max_frames', type=int, default=None, help='最大处理帧数')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='运行设备')

    args = parser.parse_args()

    try:
        # 初始化上色器
        colorizer = VideoColorizer(args.model, args.device)

        # 判断输入类型
        input_ext = os.path.splitext(args.input)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        if input_ext in image_extensions:
            # 处理图片
            colorizer.process_image(args.input, args.output)
        else:
            # 处理视频
            colorizer.process_video(
                input_path=args.input,
                output_path=args.output,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames
            )

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # 如果直接运行，使用示例配置
    if len(os.sys.argv) == 1:
        print("🚀 DeOldify 视频上色器")
        print("💡 提示: 使用命令行参数运行以获得更好体验")
        print("   示例: python complete_video_colorizer.py --model model.pth --input video.mp4 --output result.mp4")
        print()

        # 使用硬编码的配置
        config = {
            'model_path': 'checkpoint_epoch_50.pth',  # 修改为你的模型路径
            'input_video': 'dl.MP4',  # 修改为你的输入视频
            'output_video': 'colorized_dl.mp4',  # 输出视频
            'frame_interval': 1,
            'max_frames': None
        }

        try:
            # 检查文件是否存在
            if not os.path.exists(config['model_path']):
                print(f" 请先将模型文件放在: {config['model_path']}")
                exit(1)

            if not os.path.exists(config['input_video']):
                print(f" 请将输入视频放在: {config['input_video']}")
                exit(1)

            # 初始化并处理
            colorizer = VideoColorizer(config['model_path'])
            colorizer.process_video(
                input_path=config['input_video'],
                output_path=config['output_video'],
                frame_interval=config['frame_interval'],
                max_frames=config['max_frames']
            )

        except Exception as e:
            print(f" 处理失败: {e}")
    else:
        # 使用命令行参数
        exit(main())