import cv2
import random
import argparse
import numpy as np
import os
from PIL import Image
from option import parse_opt
class make_patches:

    def __init__(self, opt):
        self.opt = opt

        if opt.set_seed:
            self.seed(opt.random)

        # 이미지를 로드하여 NumPy 배열로 변환
        image = cv2.cvtColor(cv2.imread(opt.image_path), cv2.COLOR_BGR2RGB)

        # 이미지 분할 및 변환 수행
        self.processed_images = self.process_image(image, opt.row_num, opt.col_num)
        random.shuffle(self.processed_images)
        
        # 변환된 이미지 출력
        for i, processed_image in enumerate(self.processed_images):
            output_image = Image.fromarray(processed_image)
            output_image.save(f'{opt.output_filename}_{i}.jpg')
        
    def seed(self, random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)

    def split_image(self, image, row, col):
        image_height, image_width = image.shape[:2]
        patch_height = image_height // row
        patch_width = image_width // col

        patches = []
        for i in range(row):
            for j in range(col):
                start_h = i * patch_height
                end_h = start_h + patch_height
                start_w = j * patch_width
                end_w = start_w + patch_width
                patch = image[start_h:end_h, start_w:end_w]
                patches.append(patch)

        return patches

    def random_mirror(self, image):
        if np.random.rand() < 0.5:
            image = np.fliplr(image)
        return image

    def random_flip(self, image):
        if np.random.rand() < 0.5:
            image = np.flipud(image)
        return image

    def random_rotation(self, image):
        if np.random.rand() < 0.5:
            angle = 90 * np.random.randint(1, 4)  # 90도 간격으로 랜덤하게 회전
            rotated_image = np.rot90(image, k=angle // 90)  # 이미지를 90도 단위로 회전
            return rotated_image
        return image

    def process_image(self, image, row, col):
        image_height, image_width = image.shape[:2]

        if image_height % row != 0:
            image = image[:-(image_height % row), :]
        if image_width % col != 0:
            image = image[:, :-(image_width % col)]

        patches = self.split_image(image, row, col)
        processed_patches = []
        for patch in patches:
            patch = self.random_mirror(patch)
            patch = self.random_flip(patch)
            patch = self.random_rotation(patch)
            processed_patches.append(patch)
        return processed_patches    

if __name__ == '__main__':
    
    opt = parse_opt()
    
    py_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(py_path)
    os.chdir(dir_name)

    dir = os.path.dirname(opt.output_filename)
    file_list = os.listdir(dir)
    for file_name in file_list:
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    make_patches(opt)