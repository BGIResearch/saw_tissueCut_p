import sys
import os
import time
from optparse import OptionParser

import h5py
import numpy as np
import tifffile as tifi
import cv2

def main():
    parser = OptionParser()
    parser.add_option("-s", "--stainimg", dest="stain_img", help="The input staining image file. ")
    parser.add_option("-o", "--outdir", dest="out_dir", help="The output directory. ")
    parser.add_option('-i', "--snId", dest="snId", help="SN identify of the stereo-chip. ")
    parser.add_option('-t', "--type", dest="stain_type", default="ssDNA", help="Tissue stain type, can be 'ssDNA' or 'conA'. ")
    opts, args = parser.parse_args()

    if opts.stain_img == None or opts.out_dir == None:
        sys.exit(not parser.print_help())

    img_path = opts.stain_img 
    name = os.path.basename(img_path)
    outname = opts.snId
    #imgSize = 256 ## 小图片的尺寸
    start = [int(x) for x in name.split(".")[0].split("_")[1:]]
    x_start,y_start = [start[0], start[1]]  ## 大图的左上角x坐标, 大图的左上角y坐标（PS. 这个坐标应该对应表达矩阵中的y.max()）

    # image path
    

    # output dir
    out_dir = opts.out_dir
    os.makedirs(out_dir, exist_ok=True)

    ## output file
    if opts.stain_type == 'ssDNA':
        file_name = outname + '.ssDNA.rpi'
    elif opts.stain_type == 'conA':
        file_name = outname + '.conA.rpi'
    h5_path = os.path.join(out_dir, file_name)

    t0 = time.time()
    img = cv2.imread(img_path, -1)
    t1 = time.time()
    print(f"Load image: {t1-t0:.2f} seconds.")
    
    ## save big image to pyramid
    createPyramid(img, h5_path, x_start, y_start)

def _write_attrs(gp, d):
    """ Write dict to hdf5.Group as attributes. """
    for k, v in d.items():
        gp.attrs[k] = v


def splitImage(im, imgSize, h5_path, bin_size):
    """ Split image into patches with imgSize and save to h5 file. """
    t0 = time.time()
    # get number of patches
    height, width = im.shape
    num_x = int(width/imgSize)+1
    num_y = int(height/imgSize)+1
    with h5py.File(h5_path, 'a') as out:
        group = out.require_group(f'bin_{bin_size}')        
        # write attributes
        attrs = {'sizex': width,
                 'sizey': height,
                 'XimageNumber': num_x, 
                 'YimageNumber': num_y}
        _write_attrs(group, attrs)
        # write dataset
        for x in range(0, num_x):
            for y in range(0, num_y):
                # deal with last row/column images                
                x_end = min(((x+1)*imgSize), width)
                y_end = min(((y+1)*imgSize), height)
                small_im = im[y*imgSize:y_end, x*imgSize:x_end]
                data_name = f'{x}/{y}'
                try:
                    # normal dataset creation
                    group.create_dataset(data_name, data=small_im)
                except Exception as e:
                    # if dataset already exists, replace it with new data
                    del group[data_name]
                    group.create_dataset(data_name, data=small_im)
    
    t1 = time.time()
    print(f"bin_{bin_size} split: {t1-t0:.2f} seconds")


def mergeImage(h5_path, bin_size):
    """ Merge image patches back to large image. """
    t0 = time.time()
    h5 = h5py.File(h5_path,'r')
    # get attributes
    imgSize = h5['metaInfo'].attrs['imgSize']    
    group = h5[f'bin_{bin_size}']
    width = group.attrs['sizex']
    height = group.attrs['sizey']
    # initialize image
    im = np.zeros((height, width), dtype=group['0/0'][()].dtype)
    # recontruct image
    for i in range(group.attrs['XimageNumber']):
        for j in range(group.attrs['YimageNumber']):
            small_im = group[f'{i}/{j}'][()]
            x_end = min(((i+1)*imgSize), width)
            y_end = min(((j+1)*imgSize), height)
            im[j*imgSize:y_end, i*imgSize:x_end] = small_im

    h5.close()
    t1 = time.time()
    print(f"Merge image: {t1-t0:.2f} seconds.")
    return im

def createPyramid(img, h5_path, x_start, y_start):
    """ Create image pyramid and save to h5. """
    # rotate image to orientate it the same way as heatmap
    # img = np.rot90(img, -1)  ## 旋转图片，配准后的图片应该不用旋转了
    # img = cv2.flip(img, 1)
    # img = np.rot90(img, 1)
    # get height and width
    t1 = time.time()
    imgSize = 256 #小图片的尺寸
    mag = [1,2,10,50,100,150]  ## dnb分辨率的列表
    height, width = img.shape
    #cv2.imwrite(os.path.join(out_dir,"rotate.png"), img)

    # write image metadata 
    if (os.path.exists(h5_path)):
        os.remove(h5_path)
    with h5py.File(h5_path, 'a') as h5_out:
        meta_group = h5_out.require_group('metaInfo')
        info = {'imgSize': imgSize,
                'x_start': x_start,
                'y_start': y_start,
                'sizex': width,
                'sizey': height}
        _write_attrs(meta_group, info)
    # write image pyramid of bin size
    for bin_size in mag:
        im_downsample = img[::bin_size, ::bin_size]
        splitImage(im_downsample, imgSize, h5_path, bin_size)
    t2 = time.time()
    print(f"Save h5: {t2-t1:.2f} seconds.")
    ## merge pyramid and compare with original image
    mim = mergeImage(h5_path, bin_size=mag[4])
    
    print(np.array_equal(img[::mag[4], ::mag[4]], cv2.flip(np.rot90(mim, -1),1)))

if __name__ == '__main__':
    main()
    