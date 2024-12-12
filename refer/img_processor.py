

from PIL import Image, ImageOps
import os
from torchvision import transforms

#获取文件夹下的文件名list
def getfilelist(folderpath):
    filename_list = os.listdir(folderpath)
    return filename_list

def merge_images(image1_path, image2_path, output_path, output_size=128):
    # 打开图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    # 确保两张图片的大小相同
    if img1.size != img2.size:
        raise ValueError("两张图片大小必须相同")
    
    # 计算每个小图块的大小，保持最终输出为 output_size x output_size
    tile_size = output_size // 2  # 每个小图片的目标大小
    
    # 调整输入图片大小为 tile_size x tile_size
    img1 = img1.resize((tile_size, tile_size))
    img2 = img2.resize((tile_size, tile_size))
    
    # 创建一个新的 output_size x output_size 的图片
    new_image = Image.new('RGB', (output_size, output_size))
    
    # 拼接图片
    new_image.paste(img1, (0, 0))                    # 左上角 1
    new_image.paste(img2, (tile_size, 0))            # 右上角 2
    new_image.paste(ImageOps.mirror(img2), (0, tile_size))            # 左下角 2
    new_image.paste(ImageOps.mirror(img1), (tile_size, tile_size))    # 右下角 1
    
    # 保存拼接后的图片
    new_image.save(output_path)

short_relation = {
    'father-daughter': 'fd',
    'father-son': 'fs',
    'mother-daughter': 'md',
    'mother-son': 'ms',
    'nonkin': 'nonkin'
}

def mergeI_what(relation):
    folderpath = 'data\\KinFaceW-I\\images\\'+relation
    output_folder = 'data\\KinFaceW-I\\merged_images'
    os.makedirs(output_folder, exist_ok=True)

    filename_list = getfilelist(folderpath)

    for i in range(0, len(filename_list), 2):
        if i + 1 < len(filename_list):
            image1_path = os.path.join(folderpath, filename_list[i])
            image2_path = os.path.join(folderpath, filename_list[i + 1])
            output_path = os.path.join(output_folder, short_relation[relation]+f'_{i // 2}.jpg')
            merge_images(image1_path, image2_path, output_path)
            
def mergeII_what(relation):
    folderpath = 'data\\KinFaceW-II\\images\\'+relation
    output_folder = 'data\\KinFaceW-II\\merged_images'
    os.makedirs(output_folder, exist_ok=True)

    filename_list = getfilelist(folderpath)

    for i in range(0, len(filename_list), 2):
        if i + 1 < len(filename_list):
            image1_path = os.path.join(folderpath, filename_list[i])
            image2_path = os.path.join(folderpath, filename_list[i + 1])
            output_path = os.path.join(output_folder, short_relation[relation]+f'_{i // 2}.jpg')
            merge_images(image1_path, image2_path, output_path)
            
            
# non_kin_count = 1
# # 读取 mat 文件
# mat_file_path = 'data\KinFaceW-II\meta_data\\fs_pairs.mat'  # 替换为你的 mat 文件路径
# data = scipy.io.loadmat(mat_file_path)

# # 设定原图像文件夹路径和目标文件夹路径
# original_image_folder = 'data\KinFaceW-II\images\\father-son'  # 替换为你的原图像文件路径
# target_folder = 'data\KinFaceW-II\images\\nonkin'  # 替换为你保存 non-kin 图像的文件夹路径
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)

# # 解析 mat 文件的结构
# folds = data.get('pairs')


# # 初始化图像对计数器


# # 遍历每个 fold

# for pair in folds:
#     kin_status = pair[1]  # 提取 kin/non-kin 信息
#     image1 = pair[2]
#     image2 = pair[3]
    
#     if kin_status == 0:  # 0 表示 non-kin（不相关的亲缘关系）
#         # 构建新的文件名
#         new_image1_name = f"none_{non_kin_count:03d}_1.jpg"
#         new_image2_name = f"none_{non_kin_count:03d}_2.jpg"
        
#         # 定位源文件路径
#         image1_path = os.path.join(original_image_folder, image1[0])
#         image2_path = os.path.join(original_image_folder, image2[0])

#         # 复制并重命名文件
#         shutil.copy(image1_path, os.path.join(target_folder, new_image1_name))
#         shutil.copy(image2_path, os.path.join(target_folder, new_image2_name))
        
#         # 更新计数器
#         non_kin_count += 1

# print(f"Successfully copied and renamed {non_kin_count-1} non-kin pairs.")

    




preprocess224_MaxVit = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess300_EfNetB3 = transforms.Compose([
    transforms.Resize(320, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

