import os
from ultralytics import YOLO
import time

# 1. 正确设置【文件夹路径】（末尾不要带具体文件名）
input_folder = r"...\images"  # 这是存放图片的文件夹
output_folder = r"...results"  # 推理结果保存文件夹

# 2. 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 3. 加载模型
model = YOLO(
    r"...\best.pt")

# 4. 获取文件夹下的所有图片文件（过滤非图片格式）
image_suffixes = (".png", ".jpg", ".jpeg", ".bmp")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_suffixes)]

# 5. 批量推理+计时
total_time = 0
for img_name in image_files:
    # 拼接完整图片路径
    img_path = os.path.join(input_folder, img_name)
    # 推理计时
    start_time = time.time()
    results = model(img_path, stream=False, verbose=False)
    infer_time = time.time() - start_time
    total_time += infer_time

    # 保存推理结果
    result_path = os.path.join(output_folder, img_name)
    results[0].save(result_path)
    print(f"ready：{img_name}，time：{infer_time:.4f} s，to：{result_path}")

# 输出平均推理时间
avg_infer_time = total_time / len(image_files) if image_files else 0
print(f"\nCompleted{len(image_files)}pictures，Average inference time：{avg_infer_time:.4f} s")