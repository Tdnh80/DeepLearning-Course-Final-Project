import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import time
import glob
import sys
from torchvision import transforms


# 从您的训练代码中复制必要的模型定义
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_channels=16):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class Generator(nn.Module):
    def __init__(self, scale=4, channels=3, base_channels=32, num_rrdb=8):
        super(Generator, self).__init__()
        self.scale = scale

        # 浅层特征提取
        self.conv_first = nn.Conv2d(channels, base_channels, 3, 1, 1)

        # RRDB块
        self.rrdb_blocks = nn.Sequential(*[RRDB(base_channels) for _ in range(num_rrdb)])

        # 后处理卷积
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, 1, 1)

        # 上采样模块
        upsampling = []
        for _ in range(int(torch.log2(torch.tensor(scale)).item())):
            upsampling.extend([
                nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        self.upsampling = nn.Sequential(*upsampling)

        # 重建卷积
        self.conv_last = nn.Conv2d(base_channels, channels, 3, 1, 1)

    def forward(self, x):
        # 浅层特征
        fea = self.conv_first(x)
        # RRDB块
        trunk = self.rrdb_blocks(fea)
        # 后处理
        fea = self.conv_body(trunk) + fea
        # 上采样
        fea = self.upsampling(fea)
        # 重建
        out = self.conv_last(fea)
        return out


class VideoSuperResolution:
    def __init__(self, model_path, scale=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.scale = scale

        # 初始化生成器
        self.generator = Generator(
            scale=scale,
            channels=3,
            base_channels=32,
            num_rrdb=8
        ).to(device)

        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            self.generator.load_state_dict(checkpoint)

        self.generator.eval()
        print(f"模型加载成功: {model_path}")

        # 图像预处理
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def preprocess_frame(self, frame):
        """预处理视频帧"""
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 转换为Tensor并归一化
        tensor = self.to_tensor(frame).unsqueeze(0).to(self.device)
        return tensor

    def postprocess_frame(self, tensor):
        """后处理输出张量"""
        tensor = tensor.squeeze(0).cpu()
        image = self.to_pil(tensor)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def enhance_frame(self, frame):
        """对单帧进行超分增强"""
        with torch.no_grad():
            # 预处理
            lr_tensor = self.preprocess_frame(frame)

            # 超分处理
            sr_tensor = self.generator(lr_tensor)

            # 后处理
            sr_frame = self.postprocess_frame(sr_tensor)

            return sr_frame

    def process_video(self, input_path, output_path, batch_size=1, keep_audio=True):
        """处理整个视频文件"""
        print(f"开始处理视频: {input_path}")

        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算超分后的尺寸
        new_width = width * self.scale
        new_height = height * self.scale

        print(f"输入视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps:.2f}")
        print(f"  总帧数: {total_frames}")
        print(f"  输出分辨率: {new_width}x{new_height}")

        # 创建视频写入器 - 修复拼写错误
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 修正：fourcc 不是 fource
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        # 处理帧
        frame_count = 0
        processing_times = []

        pbar = tqdm(total=total_frames, desc="处理视频帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            try:
                # 超分处理
                enhanced_frame = self.enhance_frame(frame)

                # 写入输出视频
                out.write(enhanced_frame)

                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)

                frame_count += 1
                pbar.update(1)

                # 显示进度信息
                if frame_count % 30 == 0:
                    avg_time = np.mean(processing_times[-30:])
                    remaining_frames = total_frames - frame_count
                    eta = remaining_frames * avg_time
                    pbar.set_postfix({
                        '进度': f'{frame_count}/{total_frames}',
                        '帧处理时间': f'{avg_time:.3f}s',
                        '预计剩余时间': f'{eta:.1f}s'
                    })

            except Exception as e:
                print(f"处理第 {frame_count} 帧时出错: {e}")
                continue

        # 释放资源
        cap.release()
        out.release()
        pbar.close()

        # 统计信息
        total_time = sum(processing_times)
        avg_processing_time = np.mean(processing_times)

        print(f"\n视频处理完成!")
        print(f"处理帧数: {frame_count}/{total_frames}")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均每帧处理时间: {avg_processing_time:.3f}秒")
        print(f"输出视频: {output_path}")

        # 如果需要保留音频，可以使用ffmpeg合并音频
        if keep_audio:
            self._add_audio_to_video(input_path, output_path)

    def _add_audio_to_video(self, input_video, output_video):
        """使用ffmpeg为输出视频添加原音频"""
        try:
            import subprocess

            temp_output = output_video.replace('.mp4', '_temp.mp4')
            os.rename(output_video, temp_output)

            # 使用ffmpeg合并音频
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-i', input_video,
                '-c', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                output_video
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_output)
            print("音频已成功添加到输出视频")

        except Exception as e:
            print(f"添加音频失败: {e}")
            # 恢复原文件
            if os.path.exists(temp_output):
                os.rename(temp_output, output_video)


class BatchVideoProcessor:
    """批量视频处理类"""

    def __init__(self, model_path, scale=4):
        self.sr_model = VideoSuperResolution(model_path, scale)

    def process_directory(self, input_dir, output_dir, extensions=['.mp4', '.avi', '.mov']):
        """处理目录中的所有视频文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_files = []
        for ext in extensions:
            video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

        print(f"找到 {len(video_files)} 个视频文件")

        for video_path in video_files:
            try:
                filename = os.path.basename(video_path)
                output_path = os.path.join(output_dir, f"enhanced_{filename}")

                print(f"\n处理视频: {filename}")
                self.sr_model.process_video(video_path, output_path)

            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                continue


class RealTimeSuperResolution:
    """实时视频超分处理（用于摄像头）"""

    def __init__(self, model_path, scale=2):  # 实时处理使用较小的放大倍数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = scale

        # 初始化模型
        self.generator = Generator(
            scale=scale,
            channels=3,
            base_channels=32,
            num_rrdb=8
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint)
        self.generator.eval()

        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def enhance_frame(self, frame):
        """对单帧进行超分增强"""
        with torch.no_grad():
            # 预处理
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            lr_tensor = self.to_tensor(frame).unsqueeze(0).to(self.device)

            # 超分处理
            sr_tensor = self.generator(lr_tensor)

            # 后处理
            sr_tensor = sr_tensor.squeeze(0).cpu()
            sr_image = self.to_pil(sr_tensor)
            sr_frame = cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)

            return sr_frame

    def start_realtime_processing(self, camera_id=0):
        """启动实时摄像头超分处理"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("实时超分处理已启动，按 'q' 退出，按 's' 保存当前帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 超分处理
            start_time = time.time()
            enhanced_frame = self.enhance_frame(frame)
            processing_time = time.time() - start_time

            # 显示原图和超分图
            original_resized = cv2.resize(frame, (enhanced_frame.shape[1] // 2, enhanced_frame.shape[0] // 2))
            combined = np.hstack([original_resized, enhanced_frame])

            # 添加信息文本
            fps_text = f"FPS: {1 / processing_time:.1f}" if processing_time > 0 else "FPS: Calculating"
            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Left: Original, Right: Enhanced", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.imshow('Real-Time Super Resolution', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = int(time.time())
                cv2.imwrite(f'frame_original_{timestamp}.jpg', frame)
                cv2.imwrite(f'frame_enhanced_{timestamp}.jpg', enhanced_frame)
                print("帧已保存")

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='视频超分辨率处理')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入视频路径或目录')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出路径')
    parser.add_argument('--model', '-m', type=str, required=True, help='模型权重路径')
    parser.add_argument('--scale', '-s', type=int, default=4, help='超分放大倍数')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--realtime', action='store_true', help='实时摄像头模式')

    args = parser.parse_args()

    if args.realtime:
        # 实时摄像头模式
        realtime_processor = RealTimeSuperResolution(args.model, scale=2)
        realtime_processor.start_realtime_processing()

    elif args.batch:
        # 批量处理模式
        batch_processor = BatchVideoProcessor(args.model, args.scale)
        batch_processor.process_directory(args.input, args.output)

    else:
        # 单视频处理模式
        sr_processor = VideoSuperResolution(args.model, args.scale)

        if os.path.isdir(args.input):
            # 输入是目录
            batch_processor = BatchVideoProcessor(args.model, args.scale)
            batch_processor.process_directory(args.input, args.output)
        else:
            # 输入是单个文件
            sr_processor.process_video(args.input, args.output)


# 简化的使用函数
def simple_video_super_resolution(input_video, output_video, model_path):
    """简化的视频超分函数"""

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator(scale=4, channels=3, base_channels=32, num_rrdb=8).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # 打开视频
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频设置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width * 4, height * 4))

    to_tensor = transforms.ToTensor()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # 转换为Tensor
        lr_tensor = to_tensor(pil_image).unsqueeze(0).to(device)

        # 超分处理
        with torch.no_grad():
            sr_tensor = generator(lr_tensor)

        # 转换回numpy
        sr_tensor = sr_tensor.squeeze(0).cpu()
        sr_image = transforms.ToPILImage()(sr_tensor)
        sr_frame = cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)

        # 写入输出
        out.write(sr_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")

    cap.release()
    out.release()
    print(f"视频超分完成! 输出保存至: {output_video}")


if __name__ == "__main__":
    # 如果没有命令行参数，显示使用方法
    if len(sys.argv) == 1:
        print("使用方法示例:")
        print("1. 单视频处理: python video_sr.py --input input.mp4 --output output.mp4 --model generator_final.pth")
        print(
            "2. 批量处理: python video_sr.py --input ./videos --output ./enhanced --model generator_final.pth --batch")
        print("3. 实时处理: python video_sr.py --model generator_final.pth --realtime")
        print("\n简化用法:")
        print("simple_video_super_resolution('input.mp4', 'output.mp4', 'generator_final.pth')")
    else:
        main()