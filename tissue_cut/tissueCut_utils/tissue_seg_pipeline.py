# Copyright (C) BGI-Reasearch - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by STOmics development team P_stomics_dev@genomics.cn, May 2017
#######################
# intensity seg
# network infer
#######################
import gc

import numpy as np
import os
import tifffile
import glog
import torch
import tissueCut_utils.tissue_seg_utils as util
import cv2
import tissueCut_utils.tissue_seg_net as tissue_net
from skimage import measure, exposure, dtype_limits
from tissueCut_utils.tissue_seg_utils import ToTensor
torch.set_grad_enabled(False)
np.random.seed(123)


class tissueCut(object):
    def __init__(self, path, out_path, type, deep):

        self.path = path
        self.type = type  # image type
        self.deep = deep  # segmentation method
        self.out_path = out_path
        # glog.info('image type: %s'%('ssdna' if type else 'RNA'))
        # glog.info('using method: %s'%('deep learning' if deep else 'intensity segmentation'))
        # init property
        self.img = []
        self.shape = []
        self.img_thumb = []
        self.mask_thumb = []
        self.mask = []
        self.file = []
        self.file_name = []
        self.file_ext = []

        self._preprocess_file(path)


    # parse file name
    def _preprocess_file(self, path):

        if os.path.isdir(path):
            self.path = path
            file_list = os.listdir(path)
            self.file = file_list
            self.file_name = [os.path.splitext(f)[0] for f in file_list]
            self.file_ext = [os.path.splitext(f)[1] for f in file_list]
        else:
            self.path = os.path.split(path)[0]
            self.file = [os.path.split(path)[-1]]
            self.file_name = [os.path.splitext(self.file[0])[0]]
            self.file_ext = [os.path.splitext(self.file[0])[-1]]


    # RNA image bin
    def _bin(self, img, bin_size= 200):

        kernel = np.zeros((bin_size, bin_size), dtype=np.uint8)
        kernel += 1
        img_bin = cv2.filter2D(img, -1, kernel)

        return img_bin


    def save_tissue_mask(self):

        # for idx, tissue_thumb in enumerate(self.mask_thumb):
        #     tifffile.imsave(os.path.join(self.out_path, self.file_name[idx] + r'_tissue_cut_thumb.tif'), tissue_thumb)

        for idx, tissue in enumerate(self.mask):
            tifffile.imsave(os.path.join(self.out_path, self.file_name[idx] + r'_tissue_cut.tif'), (tissue > 0).astype(np.uint8))
        glog.info('seg results saved in %s'%self.out_path)


    # preprocess image for deep learning
    def get_thumb_img(self):

        glog.info('image loading and preprocessing...')

        for ext, file in zip(self.file_ext, self.file):
            assert ext in ['.tif', '.png', '.jpg']
            if ext == '.tif':

                img = tifffile.imread(os.path.join(self.path, file))
                if len(img.shape) == 3:
                    img = img[:, :, 0]
            else:

                img = cv2.imread(os.path.join(self.path, file), 0)

            self.img.append(img)

            self.shape.append(img.shape)

            if self.deep:

                if self.type:
                    """ssdna: equalizeHist"""

                    if img.dtype != 'uint8':

                        img = util.transfer_16bit_to_8bit(img)

                    img_pre = cv2.equalizeHist(img)

                else:
                    """rna: bin 200"""
                    if(img.shape[0]<500):

                        img = self._bin(img)
                        if img.dtype != 'uint8':
                            img = util.transfer_16bit_to_8bit(img)

                        img_pre = exposure.adjust_log(img)
                    else:
                        img = self._bin(img)
                        if img.dtype != 'uint8':
                            img = self.orTransfer_16bit_to_8bit(img)
                            gc.collect()
                        img_pre = self.orAdjust_log(img)
                        # img_pre = cv2.resize(img_pre2, (imgCol,imgRow))

                img_thumb = util.down_sample(img_pre, shape=(1024, 2048))
                del img_pre
                gc.collect()
                self.img_thumb.append(img_thumb)

    def orTransfer_16bit_to_8bit(self,image_16bit):

        min_16bit = np.min(image_16bit)
        max_16bit = np.max(image_16bit)
        image_8bit=np.zeros(image_16bit.shape,dtype=np.uint8)
        for i in range(0, image_16bit.shape[0]):
            image_8bit[i] = np.array(np.rint(255 * ((image_16bit[i] - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

        return image_8bit
    def orAdjust_log(self,img):
        dtype = img.dtype.type
        scale = float(dtype_limits(img, True)[1] - dtype_limits(img, True)[0])
        out = np.zeros(img.shape, dtype=np.uint8)
        for i in range(0, img.shape[0]):
            out[i] = np.log2(1 + img[i] / scale) * scale * 1
        return out.astype(dtype)
    # infer tissue mask by network
    def tissue_infer_deep(self):
        # network infer

        self.get_thumb_img()

        # define tissueCut_model
        net = tissue_net.TissueSeg(2)

        if self.type:
            model_path = os.path.join(os.path.split(__file__)[0], '../tissueCut_model/ssdna_seg.pth')
        else:
            model_path = os.path.join(os.path.split(__file__)[0], '../tissueCut_model/rna_seg.pth')

        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        net.eval()
        # net.cuda()

        # prepare data
        to_tensor = ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115),
        )
        glog.info('tissueCut_model infer...')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        for shape, im, file, img_thumb in zip(self.shape, self.img_thumb, self.file, self.img_thumb):

            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)#.cuda()

            # inference
            out = np.array(net(im, )[0].argmax(dim=1).squeeze().detach().cpu().numpy(), dtype=np.uint8)
            out = util.hole_fill(out).astype(np.uint8)
            img_open = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

            self.mask_thumb.append(np.uint8(img_open > 0))

        self.__get_roi()


    # tissue segmentation by intensity filter
    def tissue_seg_intensity(self):

        def getArea(elem):
            return elem.area

        self.get_thumb_img()

        glog.info('segment by intensity...')
        for idx, ori_image in enumerate(self.img):
            shapes = ori_image.shape

            # downsample ori_image
            if not self.type:
                ori_image = self._bin(ori_image, 80)

            # tifffile.imsave(os.path.join(self.out_path, self.file[idx] + '_bin.tif'), ori_image)

            image_thumb = util.down_sample(ori_image, shape=(shapes[0] // 5, shapes[1] // 5))

            if image_thumb.dtype != 'uint8':
                image_thumb = util.transfer_16bit_to_8bit(image_thumb)

            self.img_thumb.append(image_thumb)

            # binary
            ret1, mask_thumb = cv2.threshold(image_thumb, 125, 255, cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  # 椭圆结构
            mask_thumb = cv2.morphologyEx(mask_thumb, cv2.MORPH_CLOSE, kernel, iterations=8)

            # choose tissue prop
            label_image = measure.label(mask_thumb, connectivity=2)
            props = measure.regionprops(label_image, intensity_image=mask_thumb)
            props.sort(key=getArea, reverse=True)
            areas = [p['area'] for p in props]
            if np.std(areas) * 10 < np.mean(areas):
                label_num = len(areas)
            else:
                label_num = int(np.sum(areas >= np.mean(areas)))
            result = np.zeros((image_thumb.shape)).astype(np.uint8)
            for i in range(label_num):
                prop = props[i]
                result += np.where(label_image != prop.label, 0, 1).astype(np.uint8)

            result_thumb = util.hole_fill(result)
            result_thumb = cv2.dilate(result_thumb, kernel, iterations=10)

            self.mask_thumb.append(np.uint8(result_thumb > 0))

        self.__get_roi()


    # filter noise tissue
    def __filter_roi(self, props):

        filtered_props= []
        for id, p in enumerate(props):

            black = np.sum(p['intensity_image'] == 0)
            sum = p['bbox_area']
            ratio_black = black / sum
            pixel_light_sum = np.sum(np.unique(p['intensity_image']) > 128)
            if ratio_black < 0.75 and pixel_light_sum > 20:
                filtered_props.append(p)
        return filtered_props


    def __get_roi(self):

        """get tissue area from ssdna"""
        for idx, tissue_mask in enumerate(self.mask_thumb):

            label_image = measure.label(tissue_mask, connectivity=2)
            props = measure.regionprops(label_image, intensity_image=self.img_thumb[idx])

            # remove noise tissue mask
            filtered_props = self.__filter_roi(props)
            if len(props) != len(filtered_props):
                tissue_mask_filter = np.zeros((tissue_mask.shape), dtype=np.uint8)
                for tissue_tile in filtered_props:
                    bbox = tissue_tile['bbox']
                    tissue_mask_filter[bbox[0]: bbox[2], bbox[1]: bbox[3]] += tissue_tile['image']
                self.mask_thumb[idx] = np.uint8(tissue_mask_filter > 0)

            if self.deep:

                ratio = int(self.img[idx].shape[0] // self.mask_thumb[idx].shape[0] // 5)

                if ratio == 0: ratio = 1
                if ratio > 20: ratio = 20
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20 // ratio, 20 // ratio))

                self.mask_thumb[idx] = cv2.dilate(self.mask_thumb[idx], kernel, iterations=10)
            self.mask.append(util.up_sample(self.mask_thumb[idx], self.img[idx].shape))


    def tissue_seg(self):

        if self.deep:
            self.tissue_infer_deep()

        else:
            self.tissue_seg_intensity()

        self.save_tissue_mask()
        del self.img
        del self.img_thumb
        return self.mask





