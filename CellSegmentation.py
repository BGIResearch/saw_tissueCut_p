import cv2
import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot as plt
import os, sys
import scipy.signal as signal
import numpy as np
import pandas as pd
import gzip

from ImageTools import CreatImg, readDNB
from CellStatAnalysis import cellStatAnalysis

class cellSegmentation():
    def __init__(self, infile, dnbFile, outpath, binSize, snId, minArea, maxArea):
        self.typeColumn = {"geneID": 'category', "x": np.uint32, "y": np.uint32, "values": np.uint32, "UMICount": np.uint32, "MIDCount": np.uint32}
        self.genedf = pd.read_csv(infile, sep='\t', dtype=self.typeColumn)
        self.dnbFile = dnbFile
        self.outpath = outpath
        self.binSize = binSize
        self.snId = snId
        self.minArea = minArea
        self.maxArea = maxArea
        self.x1 ,self.x2, self.y1, self.y2 = self.genedf['x'].min(), self.genedf['x'].max(), self.genedf['y'].min(), self.genedf['y'].max()

    def process(self):
        print("Creating bin image..")
        binImg = CreatImg(self.genedf, self.binSize)
        offsetX, offsetY, binImg = self.DnbMatting(binImg) 
        img_data = self.ImgSplit(binImg)

        print("Watershed split cells..")
        watershed_mask = []
        for img in img_data:
            blurred = cv2.medianBlur(img, 3)
            local_eql_img = self.Local_equalize(blurred)
        
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opening = cv2.morphologyEx(local_eql_img, cv2.MORPH_OPEN, kernel)

            peaks = self.CalHist(local_eql_img)
            _, thresh = cv2.threshold(opening, peaks[-1], 255, cv2.THRESH_BINARY)

            tmpImg = img.copy()
            mask = self.watershed_mask(thresh, tmpImg)
            watershed_mask.append(mask)
    
        watershed_img = self.ImgCombine(watershed_mask)
        cnts, hie = cv2.findContours(watershed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Total contour detected: ", len(cnts))

        print("Create mask and label cells..")
        label_mask, ori_cnts, labeldf = self.cellSegmentation(cnts, hie, binImg)
        ori_cnts = np.array(ori_cnts)

        if len(ori_cnts) == 0:
            self.binSize = self.binSize * 2
            print("Failed to detect cell at binSize={}, try binSize={}".format(self.binSize, 2*self.binSize))
            binImg = CreatImg(self.genedf, self.binSize)
            img_data = self.ImgSplit(binImg)

            print("Watershed split cells..")
            watershed_mask = []
            for img in img_data:
                blurred = cv2.medianBlur(img, 3)
                local_eql_img = self.Local_equalize(blurred)
            
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                opening = cv2.morphologyEx(local_eql_img, cv2.MORPH_OPEN, kernel)

                peaks = self.CalHist(local_eql_img)
                _, thresh = cv2.threshold(opening, peaks[-1], 255, cv2.THRESH_BINARY)

                tmpImg = img.copy()
                mask = self.watershed_mask(thresh, tmpImg)
                watershed_mask.append(mask)
        
            watershed_img = self.ImgCombine(watershed_mask)
            cnts, hie = cv2.findContours(watershed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print("Total contour detected: ", len(cnts))

            print("Create mask and label cells..")
            label_mask, ori_cnts, labeldf = self.cellSegmentation(cnts, hie, binImg)
            ori_cnts = np.array(ori_cnts)

            if len(ori_cnts) == 0:
                self.genedf.to_csv(os.path.join(self.outpath, "merge_GetExp_gene.txt"), sep='\t', index=False)
                tot_gene_type = len(set(self.genedf['geneID']))
                logpath = os.path.join(self.outpath, "TissueCut.log")
                with open(logpath, "w") as log:
                    log.write("############## Cell Statistic Analysis ############\n")
                    log.write("Total_contour_area: {}\n".format(0))
                    log.write("Number_of_DNB_under_cell: {}\nRatio: {:.2f}\n".format(0, (0)*100))
                    log.write("Total_Gene_type: {}\n".format(tot_gene_type))
                    log.write("Total_umi_under_cell: %d\n" %(0))
                    log.write("Reads_under_cell: {}\nFraction_Reads_in_Spots_Under_cell: {:.2f}\n".format(0, (0)*100))
                    log.write("\n")
                
                    log.write("Total_cell_count: %d\n" % (0))
                    log.write("Mean_reads: {}\n".format(0))
                    log.write("Median_reads: {}\n".format(0))
                    log.write("Mean_Gene_type_per_cell: {:.2f}\nMedian_Gene_type_per_cell: {}\n".format(0, 0))
                    log.write("Mean_Umi_per_cell: {:.2f}\nMedian_Umi_per_cell: {}\n".format(0, 0))
                    log.write("Mean_cell_area: {}\n".format(0))
                    log.write("Mean_DNB_per_cell: {:.2f}\nMedian_DNB_per_cell: {}\n".format(0, 0))
                    log.write("Mean_MTgene_counts_per_cell: {:.2f}\n".format(0))
                    log.write("Mean_RPLgene_counts_per_cell: {:.2f}\n".format(0))
                    log.write("Mean_gene_type_per_DNB: {:.2f}\n".format(0))
                    log.write("Mean_Umi_per_DNB: {:.2f}\n".format(0))
                return 0

        ### Draw cell contours
        oriImg = CreatImg(self.genedf, 1)
        # oriImg = CreatImg(cellSeg.genedf, 1)
        tmpImg = np.stack((oriImg, )*3, axis=2)
        cv2.drawContours(tmpImg, ori_cnts, -1, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(self.outpath, "cell_contours.tiff"), tmpImg)
        mergedf, tissuedf = self.MergeResult(label_mask, oriImg)
        # tissuedf.to_csv(os.path.join(outpath, "Tissue_df.txt"), sep='\t')
        mergedf.to_csv(os.path.join(self.outpath, "{0}.tissue.gem.gz".format(self.snId)), sep='\t', index=False)

        labelset = sorted(set(mergedf['label']))
        areadf = pd.DataFrame()
        areadf['label'] = labelset
        merge_area = pd.merge(labeldf, areadf, on=['label'], how='inner')
        merge_area.to_csv(os.path.join(self.outpath, "cell_stat.txt"), sep='\t', index=False)

        print("cell segmentation Done.")

        print("Merge cell reads..")
        if self.dnbFile is not None:
            dnbdf = readDNB(self.dnbFile, offsetX, offsetY)
        else:
            dnbdf = pd.DataFrame(columns=['x', 'y', 'reads'])
        mergetis = pd.merge(tissuedf, dnbdf, how='inner')
        num_dnb = len(mergetis['x'])
        tot_reads = dnbdf['reads'].sum()
        print("number of dnb: ", num_dnb, "total reads: ", tot_reads)
        # mergetis.to_csv(os.path.join(outpath, "Merge_reads.txt"), sep='\t')
    
        print("Cell statistic analysis starts..")
        cs = cellStatAnalysis(mergedf, self.outpath)
        statdf= cs.CellProcess(mergetis)
        statdf.to_csv(os.path.join(self.outpath, "GetExp_Statisticresult.detailed.txt"), sep='\t', index=False)
        if (statdf.empty):
            statdf = statdf.append({'Genetype':1,  'MTgene':1, "RPLgene":1, 'umi_counts':1, 'DNB_counts':1, 'Read_counts':1, 'GeneperDNB':1, 'UmiperDNB':1}, ignore_index=True) 
        cs.StatAnalysis(statdf, labeldf, num_dnb, tot_reads)
        # cs._CalFrequency(df)
        print("Done")

    def ImgSplit(self, img):
        ## default split into 4 small images
        img_tl = img[:img.shape[0]//2, :img.shape[1]//2]
        img_tr = img[:img.shape[0]//2, img.shape[1]//2:]
        img_dl = img[img.shape[0]//2:, :img.shape[1]//2]
        img_dr = img[img.shape[0]//2:, img.shape[1]//2:]
        img_data = [img_tl, img_tr, img_dl, img_dr]

        return img_data

    def ImgCombine(self, watershed_mask):
        watershed_img_up = np.concatenate(watershed_mask[:2], axis=1)
        watershed_img_down = np.concatenate(watershed_mask[2:], axis=1)
        watershed_img = np.concatenate([watershed_img_up, watershed_img_down], axis=0)
        return watershed_img
    
    def CalHist(self, img, fig=0):
        data = img[img > 0]
        hist = cv2.calcHist([data], [0], None, [256], [0, 255])
        frq = np.concatenate(hist)
        peaks, _ = signal.find_peaks(frq, height=1.5 * frq.mean())
        if fig:
            plt.figure()
            plt.plot(hist)
            plt.plot(peaks, frq[peaks], "X")
            plt.show()
        return peaks

    def Local_equalize(self, img, gridSize=5):
        calhe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(gridSize, gridSize))
        gray = calhe.apply(img)
        return gray

    def watershed_mask(self, opening, binImg):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(opening, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(),255,cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        
        tmpImg = np.stack((binImg, binImg, binImg), axis=2)
        markers = cv2.watershed(tmpImg,markers)
        tmpImg[markers == -1] = [0, 0, 255]
        # cv2.imwrite(os.path.join(outpath, str(tot) + "_watershed_output.tiff"), tmpImg)
        
        mask = np.zeros(binImg.shape, np.uint8)
        mask[markers == -1] = 255
        return mask

    def __Circularity(self, im):
        from skimage.measure import regionprops
        props = regionprops(im)
        return props

    def cellSegmentation(self, cnts, hie, img):
        tot = 0
        print("Binsize for now: ", self.binSize)
        ### create image to circle cells
        outImg = np.stack((img, )*3, axis=2)

        ### Create binSize=1 image
        x1, x2 = self.genedf['x'].min(), self.genedf['x'].max()
        y1, y2 = self.genedf['y'].min(), self.genedf['y'].max()
        shape = (y2 - y1 + 1, x2 - x1 + 1)
        im = np.zeros(shape, np.uint16)

        ### recover contours to bin1
        center_x = []
        center_y = []
        Area = []
        ori_cnts = []
        for i in range(len(cnts)):
            c = cnts[i]
            h = hie[0][i]
            if h[2] != -1:
                continue
            tmpcons = []
            for i in c:
                tmpcons.append([[(i[0][0])*self.binSize, (i[0][1])*self.binSize]])
            tmpcons = np.array(tmpcons)
            tmparea = cv2.contourArea(tmpcons)

            ## filter out area larger than maxArea or less than min Area
            if tmparea < self.minArea or tmparea > self.maxArea:
                continue
            
            tot += 1
            ori_cnts.append(tmpcons)
            ## label cells 
            cv2.fillPoly(im, [tmpcons], tot)

            ### compute radius and circle cell
            r = cv2.arcLength(c, True)/(2*np.pi)
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(outImg, (cx, cy), int(r),(0, 0, 255), 2)

            ### label area center
            oriM = cv2.moments(tmpcons)
            ori_cx = int(oriM['m10']/oriM['m00']) + x1
            ori_cy = int(oriM['m01']/oriM['m00']) + y1
            center_x.append(ori_cx)
            center_y.append(ori_cy)
            Area.append(tmparea)
            
        print("Total cell detected: ", tot)
        labeldf = pd.DataFrame()

        if len(ori_cnts) == 0:
            return im, ori_cnts, labeldf

        print("Saving contour info..")
        labeldf['label'] = list(range(1, tot+1))
        labeldf['x'] = center_x
        labeldf['y'] = center_y
        labeldf['CellArea'] = Area
        # labeldf.to_csv(os.path.join(self.outpath, "merge_GetExp_gene_labeled_stat.txt"), sep='\t', index=False)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilate_mask = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=2)

        cv2.imwrite(os.path.join(self.outpath, "Circled_cell.tiff"), outImg)
        return dilate_mask, ori_cnts, labeldf

    def MergeResult(self, mask, oriImg):
        tmp = oriImg.copy()
        dst = np.bitwise_and(tmp, mask)
        dstx = np.where(dst > 0)[1]
        dsty = np.where(dst > 0)[0]
        
        x1, x2 = self.genedf['x'].min(), self.genedf['x'].max()
        y1, y2 = self.genedf['y'].min(), self.genedf['y'].max()
        tissue = pd.DataFrame()
        tissue['x'] = [ii + x1 for ii in dstx]
        tissue['y'] = [ij + y1 for ij in dsty]
        tissue['label'] = mask[np.where(dst > 0)]

        mergedf = pd.merge(self.genedf, tissue, on=['x', 'y'], how='inner')
        return mergedf, tissue

    def DnbMatting(self, binImg):
        gradx = cv2.Sobel(binImg, ddepth=-1, dx=1, dy=0, ksize=-1)
        grady = cv2.Sobel(binImg, ddepth=-1, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradx, grady)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.blur(gradient, (3, 3))
        (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=10)
        closed = cv2.dilate(closed, None, iterations=10)
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #imageFile = os.path.join(self.outpath, "{0}x{0}_image.tif".format(self.binSize))
        if(len(cnts) < 1):         
            bx1 = self.x1
            by1 = self.y1
            bx2 = self.x2
            by2 = self.y2
            filterGene = self.genedf
            #cv2.imwrite(imageFile, binImg)
        else:
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            #cv2.drawContours(binImg, [box], -1, (255, 0, 0), 1)
            #cv2.imwrite(imageFile, binImg)
            Xs = box[...,0]
            Ys = box[...,1]
            minX, maxX = max(0, min(Xs) - 10), max(Xs) + 10
            minY, maxY = max(0, min(Ys) - 10), max(Ys) + 10
            cv2.drawContours(binImg, [box], -1, (255, 0, 0), 1)
            dnbMattingImgFile = os.path.join(self.outpath, "{0}X{0}_contour_image.png".format(self.binSize))
            cv2.imwrite(dnbMattingImgFile, binImg)
            bx1, bx2, by1, by2 = max(0, minX*self.binSize+self.x1), maxX*self.binSize+self.x1, max(0, minY*self.binSize+self.y1), maxY*self.binSize+self.y1
            filterGene = self.genedf.loc[(self.genedf['x']>=bx1)&(self.genedf['x']<=bx2)&(self.genedf['y']>=by1)&(self.genedf['y']<=by2)]
            bx1 = filterGene['x'].min()
            by1 = filterGene['y'].min()
        filterGene['x'] = filterGene['x'] - bx1
        filterGene['y'] = filterGene['y'] - by1
        self.genedf = filterGene
        self.x1 = 0
        self.y1 = 0
        self.x2 = bx2 - bx1
        self.y2 = by2 - by1
        binImg = binImg[by1:by2, bx1:bx2]
        self.ori_shape = (self.y2 - self.y1 + 1, self.x2 - self.x1 + 1)
        #print ("minX: {0}\tmaxX: {1}\tminY: {2}\tmaxY: {3}".format(minX, maxX, minY, maxY))
        print ("x1: {0}\tx2: {1}\ty1: {2}\ty2: {3}".format(bx1, bx2, by1, by2))
        filterGenefile = os.path.join(self.outpath, "{0}.gem.gz".format(self.snId))
        with gzip.open(filterGenefile, 'wt') as writer:
            self.info="#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip={0}\n#OffsetX={1}\n#OffsetY={2}\n".format(self.snId, bx1, by1)
            writer.write(self.info)
            filterGene[['geneID', 'x', 'y', 'MIDCount']].to_csv(writer, index=None, sep="\t")
        
        return bx1, by1, binImg
