import os
from PIL import Image
def center_crop(im): 
    crop_size = min(im.size) 
    left = (im.size[0] - crop_size)//2 
    top = (im.size[1] - crop_size)//2 
    right = (im.size[0] + crop_size)//2 
    bottom = (im.size[1] + crop_size)//2 

    return im.crop((left, top, right, bottom))

def convert_image(src_file, dest_file):
    _,ext = os.path.splitext(src_file)
    if ext != '.JPG':
        return

    src_image = Image.open(src_file)
    dest_image = src_image.convert("L")
    dest_image = center_crop(dest_image).resize((32,32))
    dest_image.save(dest_file)

base_dir = "C:/Users/student/Face_Recog/face_image/" #기존폴더
target_dir= "C:/Users/student/Face_Recog/face_image_processing/" #이미지 프로세싱 폴더

list = os.listdir(base_dir)
for dir in list :                   #상위 디렉토리 순회
    dest = target_dir + dir
    if not os.path.exists(dest):
        os.makedirs(dest)

    src_dir = base_dir + dir        #기존 폴더/하위 (ex.Images/00000)
    file_list = os.listdir(src_dir)
    
    for file in file_list :         #개별이미지 디렉토리 순회
        src_file = src_dir + "/" + file
        dest_file = dest + "/" +file
        convert_image(src_file, dest_file)

    print(src_dir, "완료")