import os

IMAGE_DIR = "./dataset/VOC2028/JPEGImages"

def rename_jpg_extensions(directory):
    renamed_count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.JPG'):
            old_path = os.path.join(directory, filename)
            
            new_filename = filename[:-4] + '.jpg'
            new_path = os.path.join(directory, new_filename)
            
            os.rename(old_path, new_path)
            print(f"重命名：{filename} -> {new_filename}")
            renamed_count += 1
    
    print(f"完成！共重命名 {renamed_count} 个文件")

rename_jpg_extensions(IMAGE_DIR)