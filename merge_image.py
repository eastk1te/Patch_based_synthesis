from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import cv2
import argparse
import numpy as np
from option import parse_opt
from PIL import Image
import random

class merge_patches:

    def __init__(self, opt, patches):
        self.opt = opt
        self.patches = patches
        
        if opt.set_seed:
            self.seed(opt.random)

        image_size = cv2.imread(opt.image_path).shape

        height = image_size[0] // opt.row_num
        width = image_size[1] // opt.col_num

        self.patches = [self.process_image(img, (height, width)) for img in self.patches]

        col_combined = self.col_combine(self.patches, opt.row_num, opt.col_num)
        result = self.row_combine(col_combined)
        output_image = Image.fromarray(result)
        output_image.save(f'{opt.result_filename}.jpg')
    
    def seed(self, random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)

    # 세로 이미지들을 가로 이미지들로 rotation
    def rotate_image(self, image):
        pil_image = Image.fromarray(image)
        rotated_image = pil_image.rotate(90, expand=True)
        rotated_image = np.array(rotated_image)
        return rotated_image

    def process_image(self, image, shape):
        if image.shape[:2] != shape:
            image = self.rotate_image(image)
        return image

    def mirror(self, image):
        image = np.fliplr(image)
        return image

    def flip(self, image):
        image = np.flipud(image)
        return image
    
    def calculate_sim(self, image1, image2):
        image1 = image1.astype(np.float64) / 255.0
        image2 = image2.astype(np.float64) / 255.0
        
        # Calculate SSIM
        ssim_score1 = ssim(image1, image2, win_size=3,data_range=1, multichannel=True)
        

        '''
        hist1 = cv2.calcHist([image1], [0], None, [256], [0,256])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0,256])

        # hist
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

        # Calculate SSIM
        ssim_score2 = ssim(image1, image2, win_size=3,data_range=1, multichannel=True, full=True)[0]

        # psnr
        psnr_score = psnr(image1, image2)

        # mse1
        mse_score = mse(image1, image2)

        # mse2
        mean_pixels = np.mean(image1, axis=0).reshape(1,-1)
        # pad_image1 = np.concatenate((mean_pixels, image1, mean_pixels), axis=0)
        pad_image2 = np.concatenate((mean_pixels, image2, mean_pixels), axis=0)
        diff_top = np.mean((image1 - pad_image2[:-2, :]) ** 2)
        diff_bottom = np.mean((image1 - pad_image2[2:, :]) ** 2)
        diff_center = np.mean((image1 - image2) ** 2)

        padd_mse_score = np.mean([diff_top, diff_bottom, diff_center])
        
        # cosine
        cos_score = cosine_similarity(image1.reshape(-1).reshape(1, -1), image2.reshape(-1).reshape(1, -1))[0][0]

        # diff
        diff = cv2.absdiff(image1, image2)

        
        
        # 에지 검출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)  # 임계값 조정 가능
        '''
        
        return ssim_score1

    # 이미지 패치들을 받음
    # 첫 이미지에서 


    def col_combine(self, image_patches, row_num, col_num):

        combined_image = []

        for _ in range(row_num):
            root_image = image_patches.pop(0)

            for _ in range(col_num-1):
                best_sim = -1
                best_idx = -1
                best_patch = None
                way = None

                for idx, patche in enumerate(image_patches):
                    
                    left = root_image[:,0,:]
                    right = root_image[:,-1,:]

                    le = patche[:,0,:] #left_edge 
                    lfe = self.flip(patche[:,0,:]) # left_flip_edge 
                    re = patche[:,-1,:] # right_edge 
                    rfe = self.flip(patche[:,-1,:]) # right_flip_edge 

                    le_le = self.calculate_sim(left, le)
                    le_lfe = self.calculate_sim(left, lfe)
                    le_re = self.calculate_sim(left, re)
                    le_rfe = self.calculate_sim(left, rfe)
                    rg_le = self.calculate_sim(right, le)
                    rg_lfe = self.calculate_sim(right, lfe)
                    rg_re = self.calculate_sim(right, re)
                    rg_rfe = self.calculate_sim(right, rfe)

                    # 왼쪽에 붙이는 경우
                    if best_sim < le_le:
                        best_sim = le_le
                        best_patch = self.mirror(patche)
                        best_idx = idx
                        way = 'left'
                    if best_sim < le_lfe:
                        best_sim = le_lfe
                        best_patch = self.mirror(self.flip(patche))
                        best_idx = idx
                        way = 'left'
                    if best_sim < le_re:
                        best_sim = le_re
                        best_patch = patche
                        best_idx = idx
                        way = 'left'
                    if best_sim < le_rfe:
                        best_sim = le_rfe
                        best_patch = self.flip(patche)
                        best_idx = idx
                        way = 'left'

                    # 오른쪽에 붙이는 경우
                    if best_sim < rg_le:
                        best_sim = rg_le
                        best_patch = patche
                        best_idx = idx
                        way = 'right'
                    if best_sim < rg_lfe:
                        best_sim = rg_lfe
                        best_patch = self.flip(patche)
                        best_idx = idx
                        way = 'right'
                    if best_sim < rg_re:
                        best_sim = rg_re
                        best_patch = self.mirror(patche)
                        best_idx = idx
                        way = 'right'
                    if best_sim < rg_rfe:
                        best_sim = rg_rfe
                        best_patch = self.mirror(self.flip(patche))
                        best_idx = idx
                        way = 'right'

                image_patches.pop(best_idx)

                if way == 'left':
                    root_image = np.concatenate((best_patch, root_image), axis=1)
                else:
                    root_image = np.concatenate((root_image, best_patch), axis=1)

            combined_image.append(root_image)
        return combined_image

    def row_combine(self, image_patches):
        root_image = image_patches.pop(0)

        for _ in range(len(image_patches)):

            best_sim = -1
            best_idx = -1
            best_patch = None
            way = None

            for idx, patche in enumerate(image_patches):
                
                up = root_image[0,:,:]
                down= root_image[-1,:,:]

                ue = patche[0,:,:] #up_edge 
                ume = self.mirror(patche)[0,:,:] # up_mirror_edge 
                de = patche[-1,:,:] # down_edge 
                dme = self.mirror(patche)[-1,:,:] # down_mirror_edge 

                # display({
                #     'text/plain': plt.imshow(root_image),
                #     'image/png': plt.show()
                # })
                # break
                
                # 위쪽에 붙이는 경우
                up_ue = self.calculate_sim(up, ue)
                up_ume = self.calculate_sim(up, ume)
                up_de = self.calculate_sim(up, de)
                up_dme = self.calculate_sim(up, dme)
                dw_ue = self.calculate_sim(down, ue)
                dw_ume = self.calculate_sim(down, ume)
                dw_de = self.calculate_sim(down, de)
                dw_dme = self.calculate_sim(down, dme)
                
                if best_sim < up_ue:
                    best_sim = up_ue
                    best_patch = self.flip(patche)
                    best_idx = idx
                    way = 'up'
                    
                if best_sim < up_ume:
                    best_sim = up_ume
                    best_patch = self.mirror(self.flip(patche))
                    best_idx = idx
                    way = 'up'
                if best_sim < up_de:
                    best_sim = up_de
                    best_patch = patche
                    best_idx = idx
                    way = 'up'
                if best_sim < up_dme:
                    best_sim = up_dme
                    best_patch = self.mirror(patche)
                    best_idx = idx
                    way = 'up'

                # 아래쪽에 붙이는 경우
                if best_sim < dw_ue:
                    best_sim = dw_ue
                    best_patch = patche
                    best_idx = idx
                    way = 'down'
                if best_sim < dw_ume:
                    best_sim = dw_ume
                    best_patch = self.mirror(patche)
                    best_idx = idx
                    way = 'down'
                if best_sim < dw_de:
                    best_sim = dw_de
                    best_patch = self.flip(patche)
                    best_idx = idx
                    way = 'down'
                if best_sim < dw_dme:
                    best_sim = dw_dme
                    best_patch = self.mirror(self.flip(patche))
                    best_idx = idx
                    way = 'down'

            image_patches.pop(best_idx)

            if way == 'up':
                root_image = np.concatenate((best_patch,root_image), axis=0)
            else:
                root_image = np.concatenate((root_image,best_patch), axis=0)
            
        return root_image
    
if __name__ == '__main__':
    
    opt = parse_opt()

    py_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(py_path)
    os.chdir(dir_name)

    image_patches = [cv2.cvtColor(cv2.imread(f'{opt.output_filename}_{i}.jpg'), cv2.COLOR_BGR2RGB) for i in range(opt.row_num * opt.col_num)]
    merge_patches(opt, image_patches)