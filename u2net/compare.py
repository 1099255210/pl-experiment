import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

base_folder = './test_data'

# 读取三个文件夹中的图片
folder_names = ['poster_c1', 'poster_c2', 'poster_c3']

images = []
for folder in folder_names:
    folder = os.path.join(base_folder, folder)
    folder_images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):  # 只处理PNG格式的图片
            image_path = os.path.join(folder, filename)
            image = imread(image_path)
            folder_images.append(image)
    images.append(folder_images)

# 绘制对比图
fig, axs = plt.subplots(len(images), len(images[0]), figsize=(20, 10))
for i, folder_images in enumerate(images):
    for j, image in enumerate(folder_images):
        axs[i, j].imshow(image)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].set_xlabel(folder_names[i])
        axs[i, j].set_ylabel(os.path.splitext(os.path.basename(image_path))[0])
plt.tight_layout()
plt.savefig('result.png')
