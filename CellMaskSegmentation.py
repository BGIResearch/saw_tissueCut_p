# -*- coding: utf-8 -*-

'''
 # @ Author: Xiaoxuan Tang
 # @ Create Time: 2021-04-25 11:24:45
 # @ Modified by: Xiaoxuan Tang
 # @ Modified time: 2021-04-27 16:49:12
 # @ Description:
 '''

import pandas as pd
import numpy as np
import cv2
import os, sys
import tifffile as tifi
import geojson
from optparse import OptionParser

class MaskSegmentation():
    def __init__(self, genedf, outpath):
        self.typeColumn = {"geneID": 'str', "x": np.uint32, "y": np.uint32, "values": np.uint32, "UMICount": np.uint32, "MIDCount": np.uint32}
        self.genedf = genedf
        self.x1, self.x2 = self.genedf['x'].min(), self.genedf['x'].max()
        self.y1, self.y2 = self.genedf['y'].min(), self.genedf['y'].max()
        self.ori_shape = (self.x2 - self.x1 + 1, self.y2 - self.y1 + 1)
        self.outpath = outpath

    def Imgsplit(self, binImg, n):
        """
        将图片拆分为n*n的小图
        """

        step_0 = binImg.shape[0] // n
        step_1 = binImg.shape[1] // n
        img_data = []
        
        for i in range(n):
            for j in range(n):
                img = binImg[i*step_0 : (i+1)*step_0, j*step_1:(j+1)*step_1]
                img_data.append(img)
        return img_data

    def Imgcombine(self, img_data, n):
        """
        图片合并
        输入为拆分后的list
        """

        mask = []
        for i in range(n):
            tmp = np.concatenate(img_data[i*n:(i+1)*n], axis=1)
            mask.append(tmp)
        Imgmask = np.concatenate(mask, axis=0)
        return Imgmask

    def read_Geojson(self, geoFile):
        """
        读取geojson，并转化为bit8的mask
        """

        mask = np.zeros(self.ori_shape, np.uint8)
        with open(geoFile, "r") as geofile:
            gj = geojson.load(geofile)
        
        for i in gj['geometries']:
            cv2.fillPoly(mask, np.array(i["coordinates"]), 255)
        
        return mask

    def convertMask(self, maskFile, flip_code):
        """
        将二值mask，转为按细胞编号的32bit/64bit的label
        """

        rotImg = np.rot90(maskFile)
        maskImg = cv2.flip(rotImg, flip_code) 
        _, labels = cv2.connectedComponents(maskImg)
        tifi.imwrite(os.path.join(self.outpath, "cell_mask.tif"), labels)

        return labels
        
    def Dumpresult(self, mask):
        """
        Merge 表达矩阵与细胞label
        """

        print("Dumping results...")
        tissuedf = pd.DataFrame()
        dst = np.nonzero(mask)

        tissuedf['x'] = dst[1] + self.x1
        tissuedf['y'] = dst[0] + self.y1
        tissuedf['label'] = mask[dst]

        res = pd.merge(self.genedf, tissuedf, on=['x', 'y'], how='inner')

        res.to_csv(os.path.join(self.outpath, "Cell_GetExp_gene.txt"), sep='\t', index=False)

    def run_cellMask(self, maskFile, flip_code):
        labels = self.convertMask(maskFile, flip_code)
        self.Dumpresult(labels)

def main():
    Usage = """
    %prog 
    -i <Gene expression matrix> 
    -m <Mask/Geojson File> 
    -o <output Path>

    return gene expression matrix under cells with labels
    """
    parser = OptionParser(Usage)
    parser.add_option("-i", dest="geneFile", help="Input gene expression matrix. ")
    parser.add_option("-o", dest="outpath", help="Output directory. ")
    parser.add_option("-m", dest="infile", help="Segmentation mask or geojson. ")
    parser.add_option("-f", dest="flip_code", type=int,  default=0, help="Image flip code. 0 for flip vertically, 1 for flip horizontally, -1 for both.")
    opts, args = parser.parse_args()

    if not opts.geneFile or not opts.outpath or not opts.infile:
        print("Inputs are not correct")
        sys.exit(not parser.print_usage())

    geneFile = opts.geneFile
    infile = opts.infile
    outpath = opts.outpath
    os.makedirs(outpath, exist_ok=True)

    suffix = infile.split(".")[-1]
    typeColumn = {"geneID": 'str', "x": np.uint32, "y": np.uint32, "values": np.uint32, "UMICount": np.uint32, "MIDCount": np.uint32}
    genedf = pd.read_csv(geneFile, sep="\t", dtype=typeColumn)
    seg = MaskSegmentation(genedf, outpath)
    print("Reading data..")
    if suffix.upper() == "GEOJSON":
        maskFile = seg.read_Geojson(infile)
    else:
        maskFile = cv2.imread(infile, -1)
    
    if maskFile is None:
        raise ValueError("Input file wrong.")

    seg.run_cellMask(maskFile, opts.flip_code)

if __name__ == '__main__':
    main()
