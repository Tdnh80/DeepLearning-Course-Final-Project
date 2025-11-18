import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 超参数配置
class Config:
    hr_path = r"D:\study\PythonProject\DIV2K_train_HR"
    lr_scale = 4  # 降采样比例
    patch_size = 64  # 图像块大小
    batch_size = 4
    num_epochs = 50
    lr = 1e-4
    num_workers = 2
    save_interval = 10

    # 模型参数
    base_channels = 32
    num_rrdb = 8
    growth_channels = 16


config = Config()


# 自定义数据集类
class DIV2KDataset(Dataset):
    def __init__(self, hr_path, lr_scale=4, patch_size=64, is_train=True):
        self.hr_path = hr_path
        self.lr_scale = lr_scale
        self.patch_size = patch_size
        self.is_train = is_train

        # 获取所有高清图像路径
        self.hr_images = [os.path.join(hr_path, f) for f in os.listdir(hr_path)
                          if f.endswith(('.png', '.jpg', '.jpeg'))]

        print(f"找到 {len(self.hr_images)} 张训练图像")

        # 数据增强变换
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # 加载高清图像
        hr_img = Image.open(self.hr_images[idx]).convert('RGB')

        # 确保图像足够大
        w, h = hr_img.size
        if w < self.patch_size or h < self.patch_size:
            hr_img = hr_img.resize((max(w, self.patch_size), max(h, self.patch_size)),
                                   Image.Resampling.BICUBIC)
            w, h = hr_img.size

        # 随机裁剪
        if self.is_train:
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            hr_img = hr_img.crop((left, top, left + self.patch_size, top + self.patch_size))
        else:
            # 测试时居中裁剪
            left = (w - self.patch_size) // 2
            top = (h - self.patch_size) // 2
            hr_img = hr_img.crop((left, top, left + self.patch_size, top + self.patch_size))

        # 数据增强
        if self.transform:
            hr_img = self.transform(hr_img)

        # 转换为张量
        hr_tensor = transforms.ToTensor()(hr_img)

        # 生成低分辨率图像
        lr_tensor = self.downsample(hr_tensor)

        return lr_tensor, hr_tensor

    def downsample(self, hr_tensor):
        """使用双三次插值降采样生成低分辨率图像"""
        # 将张量转换回PIL图像进行降采样
        hr_img = transforms.ToPILImage()(hr_tensor)
        w, h = hr_img.size

        # 降采样到低分辨率
        lr_w, lr_h = w // self.lr_scale, h // self.lr_scale
        lr_img = hr_img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)

        # 注意：这里不要上采样回原始大小，让生成器学习上采样
        return transforms.ToTensor()(lr_img)


# 定义残差密集块 (Residual Dense Block)
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


# 定义残差缩放密集块 (RRDB)
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


# 定义生成器 - 修复上采样尺寸问题
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

        # 上采样模块 - 修复尺寸计算
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
        # 保存输入尺寸用于验证
        input_size = x.size()

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

        # 验证输出尺寸
        expected_height = input_size[2] * self.scale
        expected_width = input_size[3] * self.scale
        if out.size(2) != expected_height or out.size(3) != expected_width:
            print(
                f"警告: 输出尺寸不匹配! 输入: {input_size[2]}x{input_size[3]}, 输出: {out.size(2)}x{out.size(3)}, 期望: {expected_height}x{expected_width}")

        return out


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, channels=3, base_channels=32):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=1, use_bn=True):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, 3, stride, 1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers.extend(discriminator_block(channels, base_channels, use_bn=False))
        layers.extend(discriminator_block(base_channels, base_channels, stride=2))
        layers.extend(discriminator_block(base_channels, base_channels * 2))
        layers.extend(discriminator_block(base_channels * 2, base_channels * 2, stride=2))
        layers.extend(discriminator_block(base_channels * 2, base_channels * 4))
        layers.extend(discriminator_block(base_channels * 4, base_channels * 4, stride=2))

        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 4, base_channels * 8, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 1, 1, 0)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


# 定义感知损失
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        # 使用更小的VGG层
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        # 调整输入尺寸以匹配VGG期望的输入
        if input.size(2) != target.size(2) or input.size(3) != target.size(3):
            # 如果尺寸不匹配，调整target尺寸
            target = nn.functional.interpolate(target, size=input.shape[2:], mode='bilinear', align_corners=False)

        # 确保输入在VGG期望的范围内
        input = (input - 0.5) / 0.5  # 从[0,1]到[-1,1]
        target = (target - 0.5) / 0.5
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.criterion(vgg_input, vgg_target)


# 初始化权重
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# 训练函数
def train_model():
    # 创建数据集和数据加载器
    dataset = DIV2KDataset(config.hr_path, config.lr_scale, config.patch_size, is_train=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, pin_memory=True)

    # 初始化模型
    generator = Generator(
        scale=config.lr_scale,
        base_channels=config.base_channels,
        num_rrdb=config.num_rrdb
    ).to(device)

    discriminator = Discriminator(base_channels=config.base_channels).to(device)

    # 打印模型参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"生成器参数数量: {count_parameters(generator):,}")
    print(f"判别器参数数量: {count_parameters(discriminator):,}")

    # 初始化权重
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # 定义损失函数
    pixel_criterion = nn.L1Loss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    vgg_criterion = VGGLoss().to(device)

    # 定义优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.9, 0.999))

    # 学习率调度器
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)

    # 训练历史
    history = {
        'g_loss': [],
        'd_loss': [],
        'pixel_loss': [],
        'adversarial_loss': [],
        'perceptual_loss': []
    }

    # 训练循环
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_pixel_loss = 0
        epoch_adversarial_loss = 0
        epoch_perceptual_loss = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')

        for i, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            batch_size = lr_imgs.size(0)

            # 打印输入尺寸用于调试
            if i == 0 and epoch == 0:
                print(f"低分辨率图像尺寸: {lr_imgs.shape}")
                print(f"高分辨率图像尺寸: {hr_imgs.shape}")

            # 真实和假的标签
            real_labels = torch.full((batch_size, 1, 1, 1), 1.0, device=device)
            fake_labels = torch.full((batch_size, 1, 1, 1), 0.0, device=device)

            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()

            # 真实图像的损失
            real_output = discriminator(hr_imgs)
            d_loss_real = adversarial_criterion(real_output, real_labels)

            # 假图像的损失
            with torch.no_grad():
                fake_imgs = generator(lr_imgs)

            # 检查尺寸是否匹配
            if fake_imgs.shape != hr_imgs.shape:
                print(f"尺寸不匹配! fake_imgs: {fake_imgs.shape}, hr_imgs: {hr_imgs.shape}")
                # 调整fake_imgs尺寸以匹配hr_imgs
                fake_imgs = nn.functional.interpolate(fake_imgs, size=hr_imgs.shape[2:],
                                                      mode='bilinear', align_corners=False)

            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = adversarial_criterion(fake_output, fake_labels)

            # 总判别器损失
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_G.zero_grad()

            # 对抗损失
            fake_output = discriminator(fake_imgs)
            g_adv_loss = adversarial_criterion(fake_output, real_labels)

            # 像素损失 - 确保尺寸匹配
            if fake_imgs.shape != hr_imgs.shape:
                fake_imgs = nn.functional.interpolate(fake_imgs, size=hr_imgs.shape[2:],
                                                      mode='bilinear', align_corners=False)

            g_pixel_loss = pixel_criterion(fake_imgs, hr_imgs)

            # 感知损失
            g_perceptual_loss = vgg_criterion(fake_imgs, hr_imgs)

            # 总生成器损失
            g_loss = g_pixel_loss + 0.1 * g_adv_loss + 0.01 * g_perceptual_loss
            g_loss.backward()
            optimizer_G.step()

            # 更新统计信息
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_pixel_loss += g_pixel_loss.item()
            epoch_adversarial_loss += g_adv_loss.item()
            epoch_perceptual_loss += g_perceptual_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'G Loss': f'{g_loss.item():.4f}',
                'D Loss': f'{d_loss.item():.4f}',
                'Pixel': f'{g_pixel_loss.item():.4f}'
            })

            # 手动垃圾回收
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 计算平均损失
        num_batches = len(dataloader)
        history['g_loss'].append(epoch_g_loss / num_batches)
        history['d_loss'].append(epoch_d_loss / num_batches)
        history['pixel_loss'].append(epoch_pixel_loss / num_batches)
        history['adversarial_loss'].append(epoch_adversarial_loss / num_batches)
        history['perceptual_loss'].append(epoch_perceptual_loss / num_batches)

        # 更新学习率
        scheduler_G.step()
        scheduler_D.step()

        # 打印epoch统计信息
        print(f'Epoch [{epoch + 1}/{config.num_epochs}] '
              f'G Loss: {history["g_loss"][-1]:.4f} '
              f'D Loss: {history["d_loss"][-1]:.4f} '
              f'Pixel Loss: {history["pixel_loss"][-1]:.4f}')

        # 保存模型
        if (epoch + 1) % config.save_interval == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')

    # 保存最终模型
    torch.save(generator.state_dict(), 'generator_final.pth')
    torch.save(discriminator.state_dict(), 'discriminator_final.pth')

    # 绘制训练历史
    plot_training_history(history)


def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['pixel_loss'], label='Pixel Loss')
    plt.title('Pixel Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['adversarial_loss'], label='Adversarial Loss')
    plt.plot(history['perceptual_loss'], label='Perceptual Loss')
    plt.title('Adversarial and Perceptual Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == '__main__':
    # 检查GPU是否可用
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 开始训练
    train_model()