# image_video_restoration_ui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from PIL import Image, ImageTk
import cv2
import numpy as np

# 导入处理类
from deoldify_video import VideoColorizer
from realesrgan_video import VideoSuperResolution, BatchVideoProcessor


class ImageVideoRestorationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI图片视频修复工具 - 上色 & 超分辨率")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f5f5f5')

        # 设置窗口图标
        try:
            self.root.iconbitmap("restoration_icon.ico")
        except:
            pass

        # 初始化变量
        self.setup_variables()

        # 创建界面
        self.create_interface()

        # 处理控制变量
        self.processing = False
        self.current_process = None

    def setup_variables(self):
        """初始化变量"""
        # 文件路径
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path_colorizer = tk.StringVar(value="deoldify_model.pth")
        self.model_path_sr = tk.StringVar(value="realesrgan_model.pth")

        # 上色参数
        self.frame_interval = tk.IntVar(value=1)
        self.max_frames = tk.IntVar(value=0)
        self.colorizer_device = tk.StringVar(value="auto")

        # 超分参数
        self.scale_factor = tk.StringVar(value="4")
        self.keep_audio = tk.BooleanVar(value=True)
        self.sr_device = tk.StringVar(value="auto")

        # 批量处理
        self.batch_input_dir = tk.StringVar()
        self.batch_output_dir = tk.StringVar()
        self.batch_file_extensions = tk.StringVar(value=".mp4,.avi,.mov,.jpg,.png")

        # 预览控制
        self.show_original = tk.BooleanVar(value=True)
        self.show_processed = tk.BooleanVar(value=True)
        self.preview_scale = tk.DoubleVar(value=0.5)

    def create_interface(self):
        """创建主界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题区域
        self.create_title_section(main_frame)

        # 创建选项卡
        self.create_notebook(main_frame)

        # 状态栏
        self.create_status_bar(main_frame)

    def create_title_section(self, parent):
        """创建标题区域"""
        title_frame = tk.Frame(parent, bg='#2c3e50', relief=tk.RAISED, bd=2)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        # 主标题
        title_label = tk.Label(
            title_frame,
            text="🛠️ AI图片视频修复工具",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white',
            pady=15
        )
        title_label.pack()

        # 副标题
        subtitle_label = tk.Label(
            title_frame,
            text="专业级黑白影像上色与超分辨率修复",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack(pady=(0, 10))

    def create_notebook(self, parent):
        """创建选项卡控件"""
        # 自定义样式
        style = ttk.Style()
        style.configure('Custom.TNotebook', background='#f5f5f5')
        style.configure('Custom.TNotebook.Tab',
                        font=('Arial', 11, 'bold'),
                        padding=[15, 5])

        notebook = ttk.Notebook(parent, style='Custom.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True)

        # 视频上色修复选项卡
        colorizer_frame = ttk.Frame(notebook, padding="15")
        self.create_colorizer_tab(colorizer_frame)
        notebook.add(colorizer_frame, text="🎨 影像上色修复")

        # 超分辨率修复选项卡
        sr_frame = ttk.Frame(notebook, padding="15")
        self.create_sr_tab(sr_frame)
        notebook.add(sr_frame, text="🔍 超分辨率修复")

        # 批量修复选项卡
        batch_frame = ttk.Frame(notebook, padding="15")
        self.create_batch_tab(batch_frame)
        notebook.add(batch_frame, text="📁 批量修复")

    def create_colorizer_tab(self, parent):
        """创建影像上色修复选项卡"""
        # 左侧配置面板
        config_frame = ttk.Frame(parent)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 右侧预览面板
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === 配置面板内容 ===
        # 模型设置
        model_frame = ttk.LabelFrame(config_frame, text="🔧 模型设置", padding="12")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="上色模型文件:").grid(row=0, column=0, sticky=tk.W, pady=6)
        ttk.Entry(model_frame, textvariable=self.model_path_colorizer, width=35).grid(row=0, column=1, padx=5, pady=6)
        ttk.Button(model_frame, text="浏览", command=self.browse_colorizer_model, width=8).grid(row=0, column=2, pady=6)

        ttk.Label(model_frame, text="运行设备:").grid(row=1, column=0, sticky=tk.W, pady=6)
        device_combo = ttk.Combobox(model_frame, textvariable=self.colorizer_device,
                                    values=["auto", "cuda", "cpu"], width=15, state="readonly")
        device_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=6, padx=5)

        # 文件选择
        file_frame = ttk.LabelFrame(config_frame, text="📁 文件选择", padding="12")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_frame, text="输入文件:").grid(row=0, column=0, sticky=tk.W, pady=6)
        ttk.Entry(file_frame, textvariable=self.input_path, width=35).grid(row=0, column=1, padx=5, pady=6)
        ttk.Button(file_frame, text="浏览", command=self.browse_input_file, width=8).grid(row=0, column=2, pady=6)

        ttk.Label(file_frame, text="输出路径:").grid(row=1, column=0, sticky=tk.W, pady=6)
        ttk.Entry(file_frame, textvariable=self.output_path, width=35).grid(row=1, column=1, padx=5, pady=6)
        ttk.Button(file_frame, text="浏览", command=self.browse_output_file, width=8).grid(row=1, column=2, pady=6)

        # 处理参数
        params_frame = ttk.LabelFrame(config_frame, text="⚙️ 处理参数", padding="12")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(params_frame, text="帧间隔:").grid(row=0, column=0, sticky=tk.W, pady=6)
        ttk.Spinbox(params_frame, from_=1, to=30, textvariable=self.frame_interval, width=8).grid(row=0, column=1,
                                                                                                  sticky=tk.W, pady=6,
                                                                                                  padx=5)
        ttk.Label(params_frame, text="(1=处理每一帧)").grid(row=0, column=2, sticky=tk.W, pady=6)

        ttk.Label(params_frame, text="最大帧数:").grid(row=1, column=0, sticky=tk.W, pady=6)
        ttk.Spinbox(params_frame, from_=0, to=100000, textvariable=self.max_frames, width=8).grid(row=1, column=1,
                                                                                                  sticky=tk.W, pady=6,
                                                                                                  padx=5)
        ttk.Label(params_frame, text="(0=无限制)").grid(row=1, column=2, sticky=tk.W, pady=6)

        # 处理控制
        control_frame = ttk.LabelFrame(config_frame, text="🎯 处理控制", padding="12")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.colorize_btn = ttk.Button(
            control_frame,
            text="开始上色修复",
            command=self.start_colorizer_processing,
            style="Accent.TButton"
        )
        self.colorize_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame,
            text="停止修复",
            command=self.stop_processing,
            state="disabled"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 信息显示
        info_frame = ttk.LabelFrame(config_frame, text="ℹ️ 修复信息", padding="12")
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.colorizer_info = scrolledtext.ScrolledText(info_frame, height=8, width=45, font=('Arial', 9))
        self.colorizer_info.pack(fill=tk.BOTH, expand=True)

        # === 预览面板内容 ===
        # 预览控制
        preview_control_frame = ttk.Frame(preview_frame)
        preview_control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(preview_control_frame, text="预览控制:").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(preview_control_frame, text="显示原图", variable=self.show_original).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(preview_control_frame, text="显示修复图", variable=self.show_processed).pack(side=tk.LEFT,
                                                                                                     padx=10)

        ttk.Label(preview_control_frame, text="缩放:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Scale(preview_control_frame, from_=0.1, to=1.0, variable=self.preview_scale,
                  orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=5)

        ttk.Button(preview_control_frame, text="加载预览", command=self.load_colorizer_preview).pack(side=tk.LEFT,
                                                                                                     padx=10)
        ttk.Button(preview_control_frame, text="清除预览", command=self.clear_preview).pack(side=tk.LEFT, padx=5)

        # 预览画布
        preview_canvas_frame = ttk.LabelFrame(preview_frame, text="👁️ 修复预览", padding="10")
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 创建对比预览区域
        comparison_frame = ttk.Frame(preview_canvas_frame)
        comparison_frame.pack(fill=tk.BOTH, expand=True)

        # 原图画布
        original_frame = ttk.Frame(comparison_frame)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(original_frame, text="原图", font=('Arial', 10, 'bold')).pack()
        self.original_canvas = tk.Canvas(original_frame, bg='white', relief=tk.SUNKEN, bd=1)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # 修复图画布
        processed_frame = ttk.Frame(comparison_frame)
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(processed_frame, text="修复结果", font=('Arial', 10, 'bold')).pack()
        self.processed_canvas = tk.Canvas(processed_frame, bg='white', relief=tk.SUNKEN, bd=1)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

    def create_sr_tab(self, parent):
        """创建超分辨率修复选项卡"""
        # 左侧配置面板
        config_frame = ttk.Frame(parent)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 右侧预览面板
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === 配置面板内容 ===
        # 模型设置
        model_frame = ttk.LabelFrame(config_frame, text="🔧 模型设置", padding="12")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="超分模型文件:").grid(row=0, column=0, sticky=tk.W, pady=6)
        ttk.Entry(model_frame, textvariable=self.model_path_sr, width=35).grid(row=0, column=1, padx=5, pady=6)
        ttk.Button(model_frame, text="浏览", command=self.browse_sr_model, width=8).grid(row=0, column=2, pady=6)

        ttk.Label(model_frame, text="运行设备:").grid(row=1, column=0, sticky=tk.W, pady=6)
        device_combo = ttk.Combobox(model_frame, textvariable=self.sr_device,
                                    values=["auto", "cuda", "cpu"], width=15, state="readonly")
        device_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=6, padx=5)

        # 文件选择
        file_frame = ttk.LabelFrame(config_frame, text="📁 文件选择", padding="12")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_frame, text="输入视频:").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.sr_input_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.sr_input_path, width=35).grid(row=0, column=1, padx=5, pady=6)
        ttk.Button(file_frame, text="浏览", command=self.browse_sr_input, width=8).grid(row=0, column=2, pady=6)

        ttk.Label(file_frame, text="输出路径:").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.sr_output_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.sr_output_path, width=35).grid(row=1, column=1, padx=5, pady=6)
        ttk.Button(file_frame, text="浏览", command=self.browse_sr_output, width=8).grid(row=1, column=2, pady=6)

        # 处理参数
        params_frame = ttk.LabelFrame(config_frame, text="⚙️ 超分参数", padding="12")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(params_frame, text="放大倍数:").grid(row=0, column=0, sticky=tk.W, pady=6)
        scale_combo = ttk.Combobox(params_frame, textvariable=self.scale_factor,
                                   values=["2", "4", "8"], width=10, state="readonly")
        scale_combo.grid(row=0, column=1, sticky=tk.W, pady=6, padx=5)

        ttk.Checkbutton(params_frame, text="保留音频", variable=self.keep_audio).grid(row=0, column=2, sticky=tk.W,
                                                                                      pady=6, padx=20)

        # 处理控制
        control_frame = ttk.LabelFrame(config_frame, text="🎯 处理控制", padding="12")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.sr_btn = ttk.Button(
            control_frame,
            text="开始超分修复",
            command=self.start_sr_processing,
            style="Accent.TButton"
        )
        self.sr_btn.pack(side=tk.LEFT, padx=5)

        self.sr_stop_btn = ttk.Button(
            control_frame,
            text="停止修复",
            command=self.stop_processing,
            state="disabled"
        )
        self.sr_stop_btn.pack(side=tk.LEFT, padx=5)

        # 信息显示
        info_frame = ttk.LabelFrame(config_frame, text="ℹ️ 修复信息", padding="12")
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.sr_info = scrolledtext.ScrolledText(info_frame, height=8, width=45, font=('Arial', 9))
        self.sr_info.pack(fill=tk.BOTH, expand=True)

        # === 预览面板内容 ===
        # 预览控制
        preview_control_frame = ttk.Frame(preview_frame)
        preview_control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(preview_control_frame, text="加载预览", command=self.load_sr_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(preview_control_frame, text="清除预览", command=self.clear_sr_preview).pack(side=tk.LEFT, padx=5)

        # 预览画布
        preview_canvas_frame = ttk.LabelFrame(preview_frame, text="👁️ 超分预览", padding="10")
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 创建对比预览区域
        comparison_frame = ttk.Frame(preview_canvas_frame)
        comparison_frame.pack(fill=tk.BOTH, expand=True)

        # 低分辨率画布
        lr_frame = ttk.Frame(comparison_frame)
        lr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(lr_frame, text="低分辨率", font=('Arial', 10, 'bold')).pack()
        self.lr_canvas = tk.Canvas(lr_frame, bg='white', relief=tk.SUNKEN, bd=1)
        self.lr_canvas.pack(fill=tk.BOTH, expand=True)

        # 高分辨率画布
        hr_frame = ttk.Frame(comparison_frame)
        hr_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(hr_frame, text="超分结果", font=('Arial', 10, 'bold')).pack()
        self.hr_canvas = tk.Canvas(hr_frame, bg='white', relief=tk.SUNKEN, bd=1)
        self.hr_canvas.pack(fill=tk.BOTH, expand=True)

    def create_batch_tab(self, parent):
        """创建批量修复选项卡"""
        # 左侧配置面板
        config_frame = ttk.Frame(parent)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 右侧日志面板
        log_frame = ttk.Frame(parent)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === 配置面板内容 ===
        # 目录设置
        dir_frame = ttk.LabelFrame(config_frame, text="📂 目录设置", padding="12")
        dir_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(dir_frame, text="输入目录:").grid(row=0, column=0, sticky=tk.W, pady=6)
        ttk.Entry(dir_frame, textvariable=self.batch_input_dir, width=35).grid(row=0, column=1, padx=5, pady=6)
        ttk.Button(dir_frame, text="浏览", command=self.browse_batch_input, width=8).grid(row=0, column=2, pady=6)

        ttk.Label(dir_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=6)
        ttk.Entry(dir_frame, textvariable=self.batch_output_dir, width=35).grid(row=1, column=1, padx=5, pady=6)
        ttk.Button(dir_frame, text="浏览", command=self.browse_batch_output, width=8).grid(row=1, column=2, pady=6)

        ttk.Label(dir_frame, text="文件扩展名:").grid(row=2, column=0, sticky=tk.W, pady=6)
        ttk.Entry(dir_frame, textvariable=self.batch_file_extensions, width=35).grid(row=2, column=1, padx=5, pady=6)

        # 处理选项
        options_frame = ttk.LabelFrame(config_frame, text="🛠️ 修复选项", padding="12")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self.batch_mode = tk.StringVar(value="colorizer")
        ttk.Radiobutton(options_frame, text="批量上色修复", variable=self.batch_mode,
                        value="colorizer").pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(options_frame, text="批量超分修复", variable=self.batch_mode,
                        value="super_resolution").pack(anchor=tk.W, pady=3)

        # 超分参数（批量）
        sr_params_frame = ttk.Frame(options_frame)
        sr_params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sr_params_frame, text="放大倍数:").pack(side=tk.LEFT, padx=(20, 5))
        self.batch_scale = tk.StringVar(value="4")
        ttk.Combobox(sr_params_frame, textvariable=self.batch_scale,
                     values=["2", "4", "8"], width=8, state="readonly").pack(side=tk.LEFT, padx=5)

        # 处理控制
        control_frame = ttk.LabelFrame(config_frame, text="🎯 批量控制", padding="12")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.batch_start_btn = ttk.Button(
            control_frame,
            text="开始批量修复",
            command=self.start_batch_processing,
            style="Accent.TButton"
        )
        self.batch_start_btn.pack(side=tk.LEFT, padx=5)

        self.batch_stop_btn = ttk.Button(
            control_frame,
            text="停止批量修复",
            command=self.stop_batch_processing,
            state="disabled"
        )
        self.batch_stop_btn.pack(side=tk.LEFT, padx=5)

        # 统计信息
        stats_frame = ttk.LabelFrame(config_frame, text="📊 统计信息", padding="12")
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        self.batch_stats = tk.Text(stats_frame, height=6, width=45, font=('Arial', 9))
        self.batch_stats.pack(fill=tk.BOTH, expand=True)

        # === 日志面板内容 ===
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(log_control_frame, text="清空日志", command=self.clear_batch_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_control_frame, text="导出日志", command=self.export_batch_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_control_frame, text="打开输出目录", command=self.open_output_directory).pack(side=tk.LEFT,
                                                                                                    padx=5)

        # 日志区域
        log_area_frame = ttk.LabelFrame(log_frame, text="📝 批量修复日志", padding="10")
        log_area_frame.pack(fill=tk.BOTH, expand=True)

        self.batch_log = scrolledtext.ScrolledText(log_area_frame, height=20, font=('Arial', 9))
        self.batch_log.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        # 状态文本
        self.status_text = tk.StringVar(value="就绪 - 欢迎使用AI图片视频修复工具")
        status_label = ttk.Label(status_frame, textvariable=self.status_text, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)

        # 进度条
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='determinate')
        progress_bar.pack(side=tk.RIGHT, padx=5, pady=2, fill=tk.X, expand=True)

    # 文件浏览方法
    def browse_colorizer_model(self):
        filename = filedialog.askopenfilename(
            title="选择上色修复模型文件",
            filetypes=[("PyTorch模型文件", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path_colorizer.set(filename)

    def browse_sr_model(self):
        filename = filedialog.askopenfilename(
            title="选择超分修复模型文件",
            filetypes=[("PyTorch模型文件", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path_sr.set(filename)

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.input_path.set(filename)
            self.update_colorizer_info(f"已选择输入文件: {os.path.basename(filename)}")

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".mp4",
            filetypes=[
                ("MP4文件", "*.mp4"),
                ("AVI文件", "*.avi"),
                ("图片文件", "*.jpg *.png"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.output_path.set(filename)

    def browse_sr_input(self):
        filename = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.sr_input_path.set(filename)
            self.update_sr_info(f"已选择输入视频: {os.path.basename(filename)}")

    def browse_sr_output(self):
        filename = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )
        if filename:
            self.sr_output_path.set(filename)

    def browse_batch_input(self):
        directory = filedialog.askdirectory(title="选择输入目录")
        if directory:
            self.batch_input_dir.set(directory)
            self.log_batch(f"输入目录设置为: {directory}")

    def browse_batch_output(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.batch_output_dir.set(directory)
            self.log_batch(f"输出目录设置为: {directory}")

    # 处理控制方法
    def start_colorizer_processing(self):
        """开始上色修复处理"""
        if not self.validate_colorizer_inputs():
            return

        self.processing = True
        self.colorize_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.update_status("开始上色修复处理...")

        thread = threading.Thread(target=self._colorizer_processing_thread)
        thread.daemon = True
        thread.start()

    def _colorizer_processing_thread(self):
        """上色修复处理线程"""
        try:
            colorizer = VideoColorizer(
                self.model_path_colorizer.get(),
                device=self.colorizer_device.get()
            )

            input_path = self.input_path.get()
            self.update_colorizer_info(f"开始处理: {os.path.basename(input_path)}")

            if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # 图片处理
                result = colorizer.process_image(input_path, self.output_path.get())
                self.update_colorizer_info("图片上色修复完成!")
            else:
                # 视频处理
                colorizer.process_video(
                    input_path=input_path,
                    output_path=self.output_path.get(),
                    frame_interval=self.frame_interval.get(),
                    max_frames=self.max_frames.get() if self.max_frames.get() > 0 else None
                )
                self.update_colorizer_info("视频上色修复完成!")

            self.update_status("上色修复处理完成!")

        except Exception as e:
            error_msg = f"上色修复失败: {str(e)}"
            self.update_colorizer_info(error_msg)
            self.update_status(error_msg)
            messagebox.showerror("修复错误", f"上色修复处理失败:\n{str(e)}")
        finally:
            self.processing = False
            self.root.after(0, self._reset_buttons)

    def start_sr_processing(self):
        """开始超分修复处理"""
        if not self.validate_sr_inputs():
            return

        self.processing = True
        self.sr_btn.config(state="disabled")
        self.sr_stop_btn.config(state="normal")
        self.update_status("开始超分修复处理...")

        thread = threading.Thread(target=self._sr_processing_thread)
        thread.daemon = True
        thread.start()

    def _sr_processing_thread(self):
        """超分修复处理线程"""
        try:
            sr_processor = VideoSuperResolution(
                self.model_path_sr.get(),
                scale=int(self.scale_factor.get())
            )

            input_path = self.sr_input_path.get()
            self.update_sr_info(f"开始处理: {os.path.basename(input_path)}")

            sr_processor.process_video(
                input_path=input_path,
                output_path=self.sr_output_path.get(),
                keep_audio=self.keep_audio.get()
            )

            self.update_sr_info("超分修复完成!")
            self.update_status("超分修复处理完成!")

        except Exception as e:
            error_msg = f"超分修复失败: {str(e)}"
            self.update_sr_info(error_msg)
            self.update_status(error_msg)
            messagebox.showerror("修复错误", f"超分修复处理失败:\n{str(e)}")
        finally:
            self.processing = False
            self.root.after(0, self._reset_buttons)

    def start_batch_processing(self):
        """开始批量修复处理"""
        if not self.validate_batch_inputs():
            return

        self.batch_processing = True
        self.batch_start_btn.config(state="disabled")
        self.batch_stop_btn.config(state="normal")
        self.log_batch("开始批量修复处理...")

        thread = threading.Thread(target=self._batch_processing_thread)
        thread.daemon = True
        thread.start()

    def _batch_processing_thread(self):
        """批量修复处理线程"""
        try:
            input_dir = self.batch_input_dir.get()
            output_dir = self.batch_output_dir.get()
            extensions = [ext.strip() for ext in self.batch_file_extensions.get().split(',')]

            if self.batch_mode.get() == "super_resolution":
                processor = BatchVideoProcessor(
                    self.model_path_sr.get(),
                    int(self.batch_scale.get())
                )
                self.log_batch(f"开始批量超分修复，放大倍数: {self.batch_scale.get()}x")
            else:
                # 这里需要实现批量上色功能
                self.log_batch("批量上色修复功能正在开发中...")
                return

            processor.process_directory(input_dir, output_dir, extensions)
            self.log_batch("批量修复完成!")

        except Exception as e:
            self.log_batch(f"批量修复失败: {str(e)}")
        finally:
            self.batch_processing = False
            self.root.after(0, self._reset_batch_buttons)

    def stop_processing(self):
        """停止处理"""
        self.processing = False
        self.update_status("处理已停止")
        self._reset_buttons()

    def stop_batch_processing(self):
        """停止批量处理"""
        self.batch_processing = False
        self.log_batch("批量处理已停止")
        self._reset_batch_buttons()

    def _reset_buttons(self):
        """重置按钮状态"""
        self.colorize_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.sr_btn.config(state="normal")
        self.sr_stop_btn.config(state="disabled")

    def _reset_batch_buttons(self):
        """重置批量处理按钮状态"""
        self.batch_start_btn.config(state="normal")
        self.batch_stop_btn.config(state="disabled")

    # 预览方法
    def load_colorizer_preview(self):
        """加载上色预览"""
        if not self.input_path.get() or not os.path.exists(self.input_path.get()):
            messagebox.showerror("错误", "请先选择输入文件")
            return

        try:
            input_path = self.input_path.get()
            if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 图片预览
                original_image = Image.open(input_path)
                self.display_image_preview(original_image, self.original_canvas)

                # 这里可以添加处理后的预览
                messagebox.showinfo("提示", "上色预览功能正在开发中")
            else:
                # 视频预览 - 提取第一帧
                cap = cv2.VideoCapture(input_path)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    original_image = Image.fromarray(frame_rgb)
                    self.display_image_preview(original_image, self.original_canvas)
                cap.release()

        except Exception as e:
            messagebox.showerror("错误", f"加载预览失败:\n{str(e)}")

    def load_sr_preview(self):
        """加载超分预览"""
        if not self.sr_input_path.get() or not os.path.exists(self.sr_input_path.get()):
            messagebox.showerror("错误", "请先选择输入视频")
            return

        try:
            # 提取视频第一帧作为预览
            cap = cv2.VideoCapture(self.sr_input_path.get())
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_image = Image.fromarray(frame_rgb)
                self.display_image_preview(original_image, self.lr_canvas)

                # 这里可以添加超分后的预览
                messagebox.showinfo("提示", "超分预览功能正在开发中")
            cap.release()

        except Exception as e:
            messagebox.showerror("错误", f"加载预览失败:\n{str(e)}")

    def display_image_preview(self, image, canvas):
        """在画布上显示图片"""
        canvas.delete("all")

        # 调整图片大小
        scale = self.preview_scale.get()
        width = int(image.width * scale)
        height = int(image.height * scale)
        resized_image = image.resize((width, height), Image.Resampling.LANCZOS)

        # 转换为PhotoImage
        photo = ImageTk.PhotoImage(resized_image)

        # 在画布中心显示图片
        canvas.image = photo  # 保持引用
        canvas.create_image(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            image=photo,
            anchor=tk.CENTER
        )

    def clear_preview(self):
        """清除上色预览"""
        self.original_canvas.delete("all")
        self.processed_canvas.delete("all")

    def clear_sr_preview(self):
        """清除超分预览"""
        self.lr_canvas.delete("all")
        self.hr_canvas.delete("all")

    # 信息更新方法
    def update_colorizer_info(self, message):
        """更新上色修复信息"""
        self.colorizer_info.insert(tk.END, f"{message}\n")
        self.colorizer_info.see(tk.END)
        self.root.update()

    def update_sr_info(self, message):
        """更新超分修复信息"""
        self.sr_info.insert(tk.END, f"{message}\n")
        self.sr_info.see(tk.END)
        self.root.update()

    def log_batch(self, message):
        """批量处理日志"""
        self.batch_log.insert(tk.END, f"{message}\n")
        self.batch_log.see(tk.END)
        self.root.update()

    def update_status(self, message):
        """更新状态"""
        self.status_text.set(message)
        self.root.update()

    def clear_batch_log(self):
        """清空批量日志"""
        self.batch_log.delete(1.0, tk.END)

    def export_batch_log(self):
        """导出批量日志"""
        filename = filedialog.asksaveasfilename(
            title="导出日志文件",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.batch_log.get(1.0, tk.END))
                messagebox.showinfo("成功", f"日志已导出到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"导出日志失败:\n{str(e)}")

    def open_output_directory(self):
        """打开输出目录"""
        if self.batch_output_dir.get() and os.path.exists(self.batch_output_dir.get()):
            os.startfile(self.batch_output_dir.get())
        else:
            messagebox.showwarning("警告", "输出目录不存在或未设置")

    # 验证方法
    def validate_colorizer_inputs(self):
        """验证上色修复输入"""
        if not self.model_path_colorizer.get() or not os.path.exists(self.model_path_colorizer.get()):
            messagebox.showerror("错误", "请选择有效的上色修复模型文件")
            return False

        if not self.input_path.get() or not os.path.exists(self.input_path.get()):
            messagebox.showerror("错误", "请选择有效的输入文件")
            return False

        if not self.output_path.get():
            messagebox.showerror("错误", "请选择输出路径")
            return False

        return True

    def validate_sr_inputs(self):
        """验证超分修复输入"""
        if not self.model_path_sr.get() or not os.path.exists(self.model_path_sr.get()):
            messagebox.showerror("错误", "请选择有效的超分修复模型文件")
            return False

        if not self.sr_input_path.get() or not os.path.exists(self.sr_input_path.get()):
            messagebox.showerror("错误", "请选择有效的输入视频")
            return False

        if not self.sr_output_path.get():
            messagebox.showerror("错误", "请选择输出路径")
            return False

        return True

    def validate_batch_inputs(self):
        """验证批量修复输入"""
        if not self.batch_input_dir.get() or not os.path.exists(self.batch_input_dir.get()):
            messagebox.showerror("错误", "请选择有效的输入目录")
            return False

        if not self.batch_output_dir.get():
            messagebox.showerror("错误", "请选择输出目录")
            return False

        return True


def main():
    """主函数"""
    # 检查依赖
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"缺少必要的依赖: {e}")
        print("请安装以下包:")
        print("pip install torch torchvision opencv-python pillow numpy")
        return

    # 创建主窗口
    root = tk.Tk()
    app = ImageVideoRestorationUI(root)

    # 设置窗口居中
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = root.winfo_reqwidth()
    window_height = root.winfo_reqheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"+{x}+{y}")

    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    main()