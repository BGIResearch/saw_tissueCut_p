import os, sys

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = "1000000000000"
import cv2
import tifffile
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters
import glob
import time
import gzip
import csv
from shutil import copyfile

import scipy.signal as signal
import seaborn as sns
import gefpy
from gefpy.bgef_writer_cy import generate_bgef
from gefpy import plot as gefplot

import h5py
from itertools import groupby

from ImageTools import CreatImg, readDNB
from ImagePyramid import createPyramid
import tissue_cut.tissueCut_utils.tissue_seg_pipeline as tissue_seg_pipeline

import gc
import glog

class tissueSegmentation():
	def __init__(self, infile, outpath, binSize, snId, develop, omics):
		self.t0 = time.time()
		self.infile = infile
		self.snId = snId
		# check input file format
		try:
			f = h5py.File(self.infile, 'r')
			f.close()
		except:
			# txt format
			# self.typeColumn = {"geneID": 'category', "x": np.uint32, "y": np.uint32, "values": np.uint32, "UMICount": np.uint32, "MIDCount": np.uint32}
			# self.genedf = pd.read_csv(infile, sep='\t', dtype=self.typeColumn, quoting=csv.QUOTE_NONE, comment="#")
			# self.resolution = [0]
			# self.version = [2]
			raise ("invalid gef format")
		# else:
		# 	self.loadGef()

		t1 = time.time()
		print("Loading data done. time used: {:.3f}".format(t1 - self.t0))
		# self.genedf['x'] = self.genedf['x'] - self.genedf['x'].min()
		# self.genedf['y'] = self.genedf['y'] - self.genedf['y'].min()
		# self.x1, self.x2 = self.genedf['x'].min(), self.genedf['x'].max()
		# self.y1, self.y2 = self.genedf['y'].min(), self.genedf['y'].max()
		# self.ori_shape = (self.y2 - self.y1 + 1, self.x2 - self.x1 + 1)
		# self.total_umi = self.genedf['MIDCount'].sum()
		self.total_umi=np.int64(0)
		self.outpath = outpath
		self.binSize = binSize
		self.figpath = os.path.join(self.outpath, "tissue_fig")
		self.stainUsed = False
		os.makedirs(self.figpath, exist_ok=True)
		self.develop = develop
		self.omics = omics

	def loadGef(self):
		f = h5py.File(self.infile, 'r')
		self.rawGefMinX = f['/geneExp/bin1/expression'].attrs['minX'][0]
		self.rawGefMinY = f['/geneExp/bin1/expression'].attrs['minY'][0]
		exp = f['/geneExp/bin1/expression']
		gene = f['/geneExp/bin1/gene']
		self.genedf = pd.DataFrame(exp[0:])
		self.genedf['x'] += self.rawGefMinX
		self.genedf['y'] += self.rawGefMinY

		genes = []
		for t in gene[0:]:
			genes.extend([t[0].decode('utf-8')] * t[2])
		self.genedf['geneID'] = np.array(genes)

		self.genedf.rename(columns={'count': 'MIDCount'}, inplace=True)
		self.genedf = self.genedf[['geneID', 'x', 'y', 'MIDCount']]

		self.resolution = exp.attrs['resolution'][0]
		self.version = f.attrs['version'][0]

		self.tissueOffsetX = 0
		self.tissueOffsetY = 0

		f.close()

	# Parse offset from SN.gef, for correction of tissue.gef
	def parseGefOffset(self, filename):
		f = h5py.File(self.infile, 'r')
		self.snOffsetX = f['/geneExp/bin1/expression'].attrs['minX'][0]
		self.snOffsetY = f['/geneExp/bin1/expression'].attrs['minY'][0]
		self.resolution = f['/geneExp/bin1/expression'].attrs['resolution']
		self.version = f.attrs['version']
		f.close()

	def process(self, stainfile, dnbfile, flip_code, multi, amp_shape, amp_factor, min_mean, platform, struct_kernel,
	            low_thresh, high_thresh):
		t1 = time.time()
		# maskTif = os.path.join(self.outpath, "bin1_mask.tif")
		# mask=tifffile.imread(maskTif)
		# glog.info('start dumpResult')
		# conArea=949038458
		# self.Dumpresult(mask, conArea)
		# # mergedf, coor, num_dnb = self.Dumpresult(mask)
		# glog.info('dumpResult done')
		# return
		ImgPath = os.path.join(self.outpath, "bin1_img.tif")
		tissueCut = tissue_seg_pipeline.tissueCut(ImgPath, self.outpath, 0, 1)
		if stainfile is None or not os.path.exists(stainfile):
			glog.info("start drawBin100imgAndcompleteGef")
			self.drawBin100imgAndcompleteGef()
			glog.info("end drawBin100imgAndcompleteGef")
			bin1Img=self.CreatImgFromFile(1,normalize=False)
			glog.info("end drawBin1imgAndcompleteGef")
			# print("Detecting contours.. ")
			# print("start createImg")
			# binImg = CreatImg(self.genedf, self.binSize)
			# cellMask = None
			# offsetX, offsetY, binImg = self.DnbMatting(binImg)
			# print("end createImg")
			# np.savetxt("bin100.txt", binImg, fmt='%d')
			# print("end save")
			# return 0
			# Img = CreatImg(self.genedf, 1, normalize=False)

			# Img = CreatImg(self.genedf, binSizeNoStain, normalize=False)

			bin1Shape=bin1Img.shape

			cv2.imwrite(ImgPath, bin1Img)
			del bin1Img
			gc.collect()
			glog.info("start to tissueCut")

			tissueCut.tissue_seg()
			gc.collect()
			glog.info("tissueCut completed")
			# print (self.ori_shape)
			# print (self.x1, self.x2, self.y1, self.y2)
			mask = tissueCut.mask[0]
			# print (mask.shape)
			# np.savetxt("bin{0}_mask.txt".format(binSizeNoStain),mask)
			maskTif=os.path.join(self.outpath, "bin1_mask.tif")
			cv2.imwrite(maskTif,mask)

			# return 0
			conArea=np.int64(0)
			# for x in range(mask.shape[0]):
			# 	for y in range(mask.shape[1]):
			# 		if mask[x][y]>0:
			# 			conArea+=1

			# conArea = len(np.where(mask > 0)[0])
			conArea=np.count_nonzero(mask)
			glog.info("conArea:\t"+str(conArea))
			if (conArea == 0):
				mask = np.ones(bin1Shape, np.uint8)
				conArea = bin1Shape[0] * bin1Shape[1]
			# cnts = self.FindContours(binImg, amp_shape, amp_factor, min_mean, platform, struct_kernel, low_thresh, high_thresh)
			# t2 = time.time()
			# print("Time used: {:.3f}".format(t2 - t1))

			# if len(cnts) == 0:
			#    mask = np.ones(self.ori_shape, np.uint8)
			#    conArea = self.ori_shape[0] * self.ori_shape[1]
			#    t3 = time.time()
			# else:
			#    print("Drawing contours...")
			#    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
			#    conArea, mask = self.CreateMask(cnt, self.binSize)
			#    tmp = np.stack((binImg,) * 3, axis=2)
			#    cv2.drawContours(tmp, [cnt], -1, (0, 0, 255), 5)
			#    cv2.imwrite(os.path.join(self.outpath, "contour_image.tif"), tmp)
			#    t3 = time.time()
			#    print("Drawing contours used: {:.3f}".format(t3 - t2))
		else:
			t2 = time.time()
			# filterGenefile = os.path.join(self.outpath, "{0}.gef".format(self.snId))
			# self.parseGefOffset(filterGenefile)
			self.stainUsed = True
			conArea, mask, cellMask, offsetX, offsetY = self.stainSegmentation(stainfile, flip_code, multi)

		# tmp = np.stack((binImg,) * 3, axis=2)
		# cv2.drawContours(tmp, [cnt], -1, (0, 0, 255), 10)
		# cv2.imwrite(os.path.join(outpath, "contour_image.tif"), tmp)
		if(conArea==0):
			print("Warning! mask is empty")
			return
		glog.info('start dumpResult')
		self.Dumpresult(mask,conArea)
		# mergedf, coor, num_dnb = self.Dumpresult(mask)
		glog.info('dumpResult done')

		#     the rest steps will be replaced with c++ code, so some data in memory can be released
		# del mergedf
		del mask
		del tissueCut
		# del self.genedf
		gc.collect()
		glog.info("start stat")
		# run c++ code
		import ctypes
		programDir = os.path.dirname(__file__)
		if programDir == '':
			programDir = './'
		tsLibPath = os.path.join(programDir, 'ts.so')
		tslib = ctypes.cdll.LoadLibrary(tsLibPath)
		tslib.argtypes = [ctypes.POINTER(ctypes.c_char)]
		if (self.omics == 'Proteomics'):
			tissueGef = os.path.join(self.outpath, "{0}.protein.tissue.gef".format(self.snId))
		else:
			tissueGef = os.path.join(self.outpath, "{0}.tissue.gef".format(self.snId))
		parameters = tissueGef
		if dnbfile is None:
			dnbfile='_VIRTUAL_FILE_PATH_'
		parameters += "," + dnbfile
		parameters += "," + str(self.tissueOffsetX)
		parameters += "," + str(self.tissueOffsetY)
		parameters += "," + str(self.snOffsetX)
		parameters += "," + str(self.snOffsetY)
		parameters += "," + str(conArea)
		parameters += "," + str(self.total_umi)
		parameters += "," + self.outpath
		parameters += "," + str(self.stainUsed)
		parameters += "," + str(self.develop)
		parameters += ",200"
		strParameters = bytes(parameters, 'utf-8')
		print(parameters)
		# string mergedf_File, string dnbFile, int offsetX, int offsetY, int coorOffsetX, int coorOffsetY, uint64_t conArea, uint64_t total_umi, string outpath, bool useStain, int maxbinSize;
		tslib.tissueStat(strParameters)
		glog.info("end stat")
		# print("Dumping result used: {:.3f}".format(t5-t3))

		# geneCount = mergedf['geneID'].nunique()
		# print("Gene type: ", geneCount)
		# umiCount = mergedf['UMICount'].sum()
		# print("Total umi: ", umiCount)
		# t6 = time.time()
		# print("Count gene and umi used: {:.3f}".format(t6 - t5))
		t5 = time.time()
		# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\tstart readDNB and merge')
		# if dnbfile != None:
		#     tmpdf = readDNB(dnbfile, offsetX, offsetY)
		#     dnbreads = tmpdf['reads'].sum()
		#     dnbdf = pd.merge(coor, tmpdf, how='inner')
		# else:
		#     dnbdf = pd.DataFrame(columns=['x', 'y', 'reads'])
		#     dnbreads = 0
		# t7 = time.time()
		# print("Load dnb file used: {:.3f}".format(t7-t5))
		# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\tstart TissueStat')
		# self.TissueStat(mergedf, dnbdf, conArea, num_dnb, dnbreads)

		# draw plot
		for b in range(50, 200 + 1, 50):
			dataFile = os.path.join(self.outpath, "bin{0}.midAndGeneCount.txt".format(b))
			my_data = np.loadtxt(dataFile)
			scapath = os.path.join(self.figpath, "scatter_{0}x{0}_MID_gene_counts.png".format(b if b != 0 else "cell"))
			violinpath = os.path.join(self.figpath, "violin_{0}x{0}_MID_gene.png".format(b if b != 0 else "cell"))
			statisticPath = os.path.join(self.figpath,
			                             "statistic_{0}x{0}_MID_gene_DNB.png".format(b if b != 0 else "cell"))
			plt.figure(figsize=(5, 5))
			# sns.scatterplot(x=df['n_counts'], y=df['n_genes'], edgecolor="gray", color="gray")
			plt.scatter(my_data[:, 0], my_data[:, 1], color="gray", edgecolors="gray", s=0.8)
			plt.grid()
			plt.xlabel("MID Count")
			plt.ylabel("Gene Number")
			plt.savefig(scapath, format="png", bbox_inches="tight")

			plt.figure(figsize=(10, 6))
			plt.subplot(121)
			sns.violinplot(y=my_data[:, 0])
			sns.stripplot(y=my_data[:, 0], jitter=0.4, color="black", size=0.8)
			plt.ylabel("")
			plt.title("MID Count")
			plt.subplot(122)
			sns.violinplot(y=my_data[:, 1])
			sns.stripplot(y=my_data[:, 1], jitter=0.4, color="black", size=0.8)
			plt.ylabel("")
			plt.title("Gene Number")
			plt.savefig(violinpath, format="png", bbox_inches="tight")
			if self.develop:
				tmpDevDf = pd.DataFrame()
				tmpDevDf['MID Count'] = my_data[:, 0]
				tmpDevDf['Gene Number'] = my_data[:, 1]
				tmpDevDf['DNB Number'] = my_data[:, 2]
				sns.set(font_scale=1.8)
				sns.set_style("whitegrid")
				g = sns.FacetGrid(pd.melt(tmpDevDf[['MID Count', 'Gene Number', 'DNB Number']]), col='variable',
				                  hue='variable', sharex=False, sharey=False, height=8, palette='Set1')
				g = (g.map(sns.distplot, "value", hist=False, rug=True))
				plt.savefig(statisticPath)
				sns.reset_defaults()
				plt.close('all')
			os.remove(dataFile)
		glog.info("end draw pictures")
		print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\tTissueStat done')
		t8 = time.time()
		# print("Get stat result used: {:.3f}".format(t8 - t7))
		print("Total segmentation time used: {:.2f}".format(t8 - self.t0))

	def AmpImg(self, image, shape, ampfactor):
		"""图像增强"""

		print("Amplification starts..")
		ampfilter = np.ones((shape, shape)) * ampfactor
		amp_fig = ndimage.convolve(image, ampfilter, mode='constant')

		print("Amplification done.")
		return amp_fig

	def CalGradient(self, binimg):
		"""计算gradient"""

		gradx = cv2.Sobel(binimg, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
		grady = cv2.Sobel(binimg, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
		# subtract the y-gradient from the x-gradient
		gradient = cv2.subtract(gradx, grady)
		gradient = cv2.convertScaleAbs(gradient)
		return gradient

	def Local_equalize(self, img, gridSize=5):
		""" 局部直方图均衡 """

		calhe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(gridSize, gridSize))
		gray = calhe.apply(img)
		return gray

	def FindThreshold(self, blurred, low_thresh, high_thresh):
		"""
        根据表达量计算过滤的阈值
        kernel: kernel size, default=10
        low, high: the min and max pixel value for finding threshold
        default low = 180, high = 255
        """

		print("Threshold reveived: ", low_thresh, high_thresh)
		# blurred = cv2.blur(gradient_img, (10, 10))
		if low_thresh and high_thresh:
			low = low_thresh
			high = high_thresh
		elif low_thresh and not high_thresh:
			low = low_thresh
			high = blurred.max()
		else:
			value = blurred[blurred > 0]
			if len(value) == 0:
				print("Failed to find threshold. exit")
				sys.exit(1)
			if not high_thresh:
				high = 255
			else:
				high = high_thresh

			hist = cv2.calcHist([value], [0], None, [256], [0, 255])
			frq = np.concatenate(hist)
			frq = signal.medfilt(frq, kernel_size=11)
			peaks, prop = signal.find_peaks(frq, width=10, height=1.5 * frq.mean())

			if len(peaks) == 0:
				low = 220
			else:
				low = min(int(peaks[-1] - prop['widths'][-1] * 0.5), 250)

			i = int(low)
			lowpath = os.path.join(self.outpath, "frequency_plot.png")
			plt.figure()
			plt.plot(frq)
			plt.plot(i, frq[i], "o")
			plt.plot(peaks, frq[peaks], "x")
			plt.savefig(lowpath)

		print("threshold has been set as: ", low, high)

		(_, thresh) = cv2.threshold(blurred, low, high, cv2.THRESH_BINARY)
		return thresh

	def FillHoles(self, thresh, struct_kernel):
		"""去噪"""

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (struct_kernel, struct_kernel))
		closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

		open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
		erodeimg = cv2.erode(closed, open_kernel, iterations=5)
		dilateimg = cv2.dilate(erodeimg, open_kernel, iterations=10)

		return dilateimg

	def CreateMask(self, cnts, binSize):
		"""生成mask"""

		contours = []
		for i in cnts:
			contours.append([[i[0][0] * binSize, i[0][1] * binSize]])

		contours = np.array(contours)
		Area = cv2.contourArea(contours)
		print("Contour area: ", Area)

		#### Smooth mask
		mask = np.zeros(self.ori_shape, np.uint8)
		epsilon = 0.01 * cv2.arcLength(contours, True)
		approx = cv2.approxPolyDP(contours, epsilon, True)
		cv2.fillPoly(mask, [approx], (255, 0, 0))
		mask = cv2.GaussianBlur(mask, (55, 55), 0)

		# cnt_mask = np.zeros(self.ori_shape, np.uint8)
		# cv2.drawContours(cnt_mask, [approx], -1, 255, 10)
		cv2.imwrite(os.path.join(self.outpath, "approx_contour_mask.tiff"), mask)
		return Area, mask

	def stainSegmentation(self, stainfile, flip_code, multi):
		"""
        根据配准图片抠图 
        输入配准好的底图，
        输出 对应可视化坐标 HE_x1_y1_height_width.png 的配准图，以及根据配准图生成的mask
        """

		if not os.path.exists(stainfile):
			print("Staining image not exist. ")
			mask = np.ones(self.ori_shape, np.uint8)
			conArea = self.ori_shape[0] * self.ori_shape[1]
			return conArea, mask, None, 0, 0
		try:
			bbox = glob.glob(stainfile + "/*tissue_bbox.csv")[0]
		except IOError:
			sys.stderr.write("bbox file dose not exists, please check...")
			sys.exit(1)

		bdf = pd.read_csv(bbox, sep="\t")
		cnt = bdf.loc(0)[0].values
		# flip image verticality
		# padding = 0
		# x1, y1, x2, y2 = [cnt[0] - padding + self.x1, cnt[1] - padding + self.y1, cnt[2] + padding + self.x1,
		#                   cnt[3] + padding + self.y1]
		#
		merPath = os.path.join(self.outpath, "{0}.gef".format(self.snId))
		# merDf = self.genedf
		# merDf = merDf.loc[(merDf['x'] >= x1) & (merDf['x'] < x2) & (merDf['y'] >= y1) & (merDf['y'] < y2)]
		#
		# merDf['x'] = merDf['x'] - x1
		# merDf['y'] = merDf['y'] - y1

		#
		# self.genedf = merDf
		# self.x1 = 0
		# self.y1 = 0
		# self.x2 = x2 - x1
		# self.y2 = y2 - y1
		# self.ori_shape = (self.y2 - self.y1 + 1, self.x2 - self.x1 + 1)
		# print("x1: {0}\tx2: {1}\ty1: {2}\ty2: {3}".format(x1, x2, y1, y2))
		#
		# self.info = "#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip={0}\n#OffsetX={1}\n#OffsetY={2}\n".format(
		# 	self.snId, x1, y1)
		# with gzip.open(merPath, 'wt') as writer:
		#    self.info="#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip={0}\n#OffsetX={1}\n#OffsetY={2}\n".format(self.snId, x1, y1)
		#    writer.write(self.info)
		#    merDf[['geneID', 'x', 'y', 'MIDCount']].to_csv(writer, index=None, sep="\t")
		# generate_bgef(self.infile, merPath,stromics=self.omics,region=[x1 - self.rawGefMinX, x2 - self.rawGefMinX, y1 - self.rawGefMinY, y2 - self.rawGefMinY])
		self.parseGefOffset(merPath)
		self.tissueOffsetX = self.snOffsetX
		self.tissueOffsetY = self.snOffsetY
		temppath = os.path.join(self.outpath, "dnb_merge")
		os.makedirs(temppath, exist_ok=True)
		gefplot.save_exp_heat_map(merPath, os.path.join(temppath, "bin200.png"))
		# generate_bgef(self.infile, merPath+".multi.gef", bin_sizes=[1,10,20,50,100,200,500])
		# merDf.to_csv(merPath, sep="\t", index=None)
		# self.x1, self.x2, self.y1, self.y2 = self.genedf['x'].min(), self.genedf['x'].max(), self.genedf['y'].min(), self.genedf['y'].max()
		# self.ori_shape = [self.y2 - self.y1, self.x2 - self.x1]
		try:
			tissueMask = glob.glob(stainfile + "/*_tissue_cut.tif")[0]
		except IndexError:
			try:
				tissueMask = glob.glob(stainfile + "/*_tissue_cut_use.tif")[0]
			except:
				sys.stderr.write("mask file for tissue cut dose not exists, please check...")
				sys.exit(1)

		try:
			pyramidImg = glob.glob(stainfile + "/*.rpi")[0]
		except IOError:
			sys.stderr.write("pyramidImg file does not exists, please check...")
			exit(1)

		try:
			cellMaskFile = glob.glob(stainfile + "/*_mask.tif")[0]
			if os.path.exists(cellMaskFile):
				cellmask = tifffile.imread(cellMaskFile)
			else:
				cellmask = None
		except IOError:
			sys.stderr.write("cell mask file dose not exists")
		except IndexError:
			cellmask = None
			sys.stderr.write("cell mask file dose not exists")

		try:
			registfile = glob.glob(stainfile + "/*_regist.tif")[0]
		except IOError:
			sys.stderr.write("regist image dose not exists, please check...")

		# stainImg = cv2.imread(registfile, -1)
		# stainImg = cv2.imread(registfile, -1)[cnt[1]: cnt[3], cnt[0]: cnt[2]]
		# stainImg = cv2.flip(stainImg, 0)
		# width, height = cnt[3] - cnt[1], cnt[2] - cnt[0]
		# cmpImg = cv2.resize(stainImg, (width // 5, height // 5), cv2.INTER_CUBIC)
		# cv2.imwrite(os.path.join(self.figpath, 'HE_{0}_{1}_{2}_{3}.png'.format(0, 0, height, width)), stainImg)

		h5OutFile = os.path.join(self.figpath, self.snId + ".ssDNA.rpi")
		try:
			copyfile(pyramidImg, h5OutFile)
		except IOError as e:
			print("Unable to copy file. %s" % e)
			exit(1)
		except:
			print("Unexpected error:", sys.exc_info())
			exit(1)

		# createPyramid(stainImg, h5OutFile, 0, 0)

		mask = tifffile.imread(tissueMask)
		# mask = cv2.imread(tissueMask, -1)
		# mask = cv2.flip(mask, 0)
		conArea = len(np.where(mask > 0)[0])
		return conArea, mask, cellmask, 0, 0

	def DnbMatting(self, binImg):
		# kernel = np.array((
		#    [1, 1, 1, 1, 1],
		#    [1, 1, 1, 1, 1],
		#    [1, 1, 1, 1, 1],
		#    [1, 1, 1, 1, 1],
		#    [1, 1, 1, 1, 1]), dtype="float32")
		# destImg = cv2.filter2D(binImg, -1, kernel)
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
		# imageFile = os.path.join(self.outpath, "{0}x{0}_image.tif".format(self.binSize))
		if (len(cnts) < 1):
			bx1 = self.x1
			by1 = self.y1
			bx2 = self.x2
			by2 = self.y2
			filterGene = self.genedf
			# cv2.imwrite(imageFile, binImg)
		else:
			c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
			rect = cv2.minAreaRect(c)
			box = np.int0(cv2.boxPoints(rect))
			# cv2.drawContours(binImg, [box], -1, (255, 0, 0), 1)
			# cv2.imwrite(imageFile, binImg)
			Xs = box[..., 0]
			Ys = box[..., 1]
			minX, maxX = max(0, min(Xs) - 10), max(Xs) + 10
			minY, maxY = max(0, min(Ys) - 10), max(Ys) + 10
			cv2.drawContours(binImg, [box], -1, (255, 0, 0), 1)
			dnbMattingImgFile = os.path.join(self.outpath, "{0}X{0}_contour_image.png".format(self.binSize))
			cv2.imwrite(dnbMattingImgFile, binImg)
			bx1, bx2, by1, by2 = max(0, minX * self.binSize + self.x1), maxX * self.binSize + self.x1, max(0,
			                                                                                               minY * self.binSize + self.y1), maxY * self.binSize + self.y1
			filterGene = self.genedf.loc[
				(self.genedf['x'] >= bx1) & (self.genedf['x'] <= bx2) & (self.genedf['y'] >= by1) & (
							self.genedf['y'] <= by2)]
			bx1 = filterGene['x'].min()
			by1 = filterGene['y'].min()
		filterGene['x'] = filterGene['x'] - bx1
		filterGene['y'] = filterGene['y'] - by1
		self.tissueOffsetX = bx1
		self.tissueOffsetY = by1
		self.genedf = filterGene
		self.x1 = 0
		self.y1 = 0
		self.x2 = bx2 - bx1
		self.y2 = by2 - by1
		binImg = binImg[by1:by2, bx1:bx2]
		self.ori_shape = (self.y2 - self.y1 + 1, self.x2 - self.x1 + 1)
		# print ("minX: {0}\tmaxX: {1}\tminY: {2}\tmaxY: {3}".format(minX, maxX, minY, maxY))
		print("x1: {0}\tx2: {1}\ty1: {2}\ty2: {3}".format(bx1, bx2, by1, by2))
		filterGenefile = os.path.join(self.outpath, "{0}.gef".format(self.snId))
		# with gzip.open(filterGenefile, 'wt') as writer:
		#    self.info="#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip={0}\n#OffsetX={1}\n#OffsetY={2}\n".format(self.snId, bx1, by1)
		#    writer.write(self.info)
		#    filterGene[['geneID', 'x', 'y', 'MIDCount']].to_csv(writer, index=None, sep="\t")
		generate_bgef(self.infile, filterGenefile,stromics=self.omics,region=[bx1 - self.rawGefMinX, bx2 - self.rawGefMinX, by1 - self.rawGefMinY,by2 - self.rawGefMinY])
		self.parseGefOffset(filterGenefile)
		temppath = os.path.join(self.outpath, "dnb_merge")
		os.makedirs(temppath, exist_ok=True)
		gefplot.save_exp_heat_map(filterGenefile, os.path.join(temppath, "bin200.png"))

		return bx1, by1, binImg

	def FindContours(self, binImg, amp_shape, amp_factor, min_mean, platform, struct_kernel, low_thresh, high_thresh):
		""" 抠图函数 """

		mean_value = binImg.mean()
		print("Image mean value: ", mean_value)
		ampImg = self.AmpImg(binImg, amp_shape, amp_factor)
		gradient = self.CalGradient(ampImg)
		blurred = cv2.blur(gradient, (11, 11))
		# thresholds = filters.threshold_isodata(blurred)
		# _, threshold = cv2.threshold(blurred, thresholds, 255, cv2.THRESH_BINARY)
		threshold = self.FindThreshold(blurred, low_thresh, high_thresh)
		# cv2.imwrite(os.path.join(self.outpath, "Threshold_image.tif"), threshold)
		if platform == 'SEQ500' or mean_value < 12:
			struct_kernel = 25
			closed = self.FillHoles(threshold, struct_kernel)
		else:
			# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
			# opening = cv2.erode(threshold, kernel)
			closed = self.FillHoles(threshold, struct_kernel)

		cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return cnts

	# Dump tissue.gef
	def dumpGef(self, outfile, df):
		with h5py.File(outfile, 'w') as f:
			# Sort by geneID, for generate /geneExp/bin1/gene
			df['x'] += self.tissueOffsetX
			df['y'] += self.tissueOffsetY
			df = df.sort_values(by='geneID').reset_index(drop=True)
			# minX = min(df['x'])
			# minY = min(df['y'])
			minX = self.snOffsetX
			minY = self.snOffsetY
			maxX = max(df['x'])
			maxY = max(df['y'])
			maxExp = max(df['MIDCount'])

			df['x'] -= minX
			df['y'] -= minY
			data = df[['x', 'y', 'MIDCount']].to_records(index=False)
			data.dtype.names = ('x', 'y', 'count')
			middt = 'u1'
			if maxExp < 255:
				middt = 'u1'
			elif maxExp < 65535:
				middt = 'u2'
			else:
				middt = 'u4'
			dt = np.dtype([('x', 'u4'), ('y', 'u4'), ('count', middt)])
			dset = f.create_dataset("geneExp/bin1/expression", data=data, dtype=dt)
			dset.attrs['minX'] = np.array([minX], dtype='u4')
			dset.attrs['minY'] = np.array([minY], dtype='u4')
			dset.attrs['maxX'] = np.array([maxX], dtype='u4')
			dset.attrs['maxY'] = np.array([maxY], dtype='u4')
			dset.attrs['maxExp'] = np.array([maxExp], dtype='u4')
			dset.attrs['resolution'] = np.array(self.resolution, dtype='u4')

			data = df[['geneID']].to_numpy()
			offset = 0
			genes = []
			for k, g in groupby(data.flat):
				gene = k
				count = sum(1 for i in g)
				genes.append((gene, offset, count))
				offset += count

			tid = h5py.h5t.C_S1.copy()
			tid.set_size(32)

			dt = np.dtype([('gene', tid), ('offset', 'u4'), ('count', 'u4')])
			data = np.array(genes, dtype=dt)
			dset = f.create_dataset("geneExp/bin1/gene", data=data)

			# add version
			f.attrs['version'] = np.array(self.version, dtype='u4')

	# @profile
	def Dumpresult(self, mask,maskLen):
		""" merge结果 """

		import time
		t0 = time.time()
		# glog.info(mask.shape)
		# write bin1 to tissue.gef
		dt = np.dtype([('x', 'u4'), ('y', 'u4'), ('count', 'u1')])


		f = h5py.File(self.infile, 'r')

		# f = h5py.File('/Users/berry/tmp/SS200000003BR_B3.tissue.gef', 'r')
		self.rawGefMinX = f['/geneExp/bin1/expression'].attrs['minX'][0]
		self.rawGefMinY = f['/geneExp/bin1/expression'].attrs['minY'][0]
		self.rawGefMaxX = f['/geneExp/bin1/expression'].attrs['maxX'][0]
		self.rawGefMaxY = f['/geneExp/bin1/expression'].attrs['maxY'][0]
		exp = f['/geneExp/bin1/expression']
		totalNum = exp.shape[0]
		existExon = False
		for k in f['/geneExp/bin1'].keys():
			if ((f['/geneExp/bin1'][k].name) == '/geneExp/bin1/exon'):
				existExon = True
				break
		if (existExon):
			exon = f['/geneExp/bin1/exon']
			exonDt=np.dtype('u1')
			exonOutData = np.zeros(totalNum, dtype=exonDt)
			# exonOutData = np.zeros(maskLen, dtype=exonDt)
		# expOutData = np.zeros(maskLen, dtype=dt)
		expOutData = np.zeros(totalNum, dtype=dt)
		batch = totalNum // 4


		iter = np.int64(0)
		totalIter=np.int64(0)
		gene = f['/geneExp/bin1/gene']
		geneNp = np.array(gene.fields(['gene', 'offset', 'count'])[0:])
		flag = np.zeros(geneNp.shape[0],dtype=np.int64)
		tmpIter=0
		for t in geneNp:
			if tmpIter == 0:
				flag[tmpIter] = t[2]
			else:
				flag[tmpIter] = t[2] + flag[tmpIter - 1]
			tmpIter+=1
		result=np.zeros(geneNp.shape[0],dtype=np.int64)
		curIdx=0

		for i in range(0, totalNum, batch):
			endIdx = totalNum if i + batch > totalNum else i + batch
			expNp = np.array(exp.fields(['x', 'y', 'count'])[i:endIdx])
			if (existExon):
				exonNp=np.array(exon[i:endIdx])
			for ele in expNp:
				self.total_umi+=ele[2]
				if(mask[ele[1]][ele[0]]>0):
					expOutData[iter]=ele
					if (existExon):
						exonOutData[iter]=exonNp[totalIter-i]
					while(totalIter>=flag[curIdx]):
						curIdx+=1
						if curIdx>=geneNp.shape[0]:
							break;
					result[curIdx]+=1

					iter += 1
				totalIter+=1
			glog.info("processed batch "+str(i//batch))
			del expNp
			if (existExon):
				del exonNp
			gc.collect()
		outGenes=[]
		offset=0
		geneCount=0
		for i in range(geneNp.shape[0]):
			if(result[i]>0):
				outGenes.append((geneNp[i][0],offset,result[i]))
				offset+=result[i]
		del geneNp
		del result
		del flag


		if(self.omics=='Proteomics'):
			tissueGef = os.path.join(self.outpath, "{0}.protein.tissue.gef".format(self.snId))
		else:
			tissueGef = os.path.join(self.outpath, "{0}.tissue.gef".format(self.snId))
		h5out = h5py.File(tissueGef,'w')

		dset=h5out.create_dataset('/geneExp/bin1/expression',data=expOutData[:iter],dtype=dt)
		if(existExon):
			if (self.omics!='Proteomics'):
				exonSet=h5out.create_dataset('/geneExp/bin1/exon',data=exonOutData[:iter],dtype=exonDt)
				maxExon=max(exonOutData[:iter])
				exonSet.attrs['maxExon']=np.array([maxExon],dtype='u4')
		minX = self.snOffsetX
		minY = self.snOffsetY
		maxX = max(expOutData['x'])
		maxY = max(expOutData['y'])
		maxExp = max(expOutData['count'])
		dset.attrs['minX'] = np.array([minX], dtype='u4')
		dset.attrs['minY'] = np.array([minY], dtype='u4')
		dset.attrs['maxX'] = np.array([maxX], dtype='u4')
		dset.attrs['maxY'] = np.array([maxY], dtype='u4')
		dset.attrs['maxExp'] = np.array([maxExp], dtype='u4')
		dset.attrs['resolution'] = np.array(self.resolution, dtype='u4')

		del expOutData
		if (existExon):
			del exonOutData
		gc.collect()

		tid = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
		tid.set_size(32)

		geneDt = np.dtype([('gene', tid), ('offset', 'u4'), ('count', 'u4')])
		geneOutNp = np.array(outGenes, dtype=geneDt)
		genedset = h5out.create_dataset("geneExp/bin1/gene", data=geneOutNp, dtype=geneDt)

		h5out.attrs['version'] = np.array(self.version, dtype='u4')
		omicsTid = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
		omicsTid.set_size(30)
		h5out.attrs['omics'] = np.array(self.omics, dtype=omicsTid)
		f.close()
		h5out.close()
		t1 = time.time()
		print("Dumping result done. time used: {:.2f}".format(t1 - t0))

		# return mergedf, coor, num_dnb

	def DumpCellresult(self, maskImg, dnbdf):
		""" 细胞分割结果 """
		t0 = time.time()
		"""
        将二值mask，转为按细胞编号的32bit/64bit的label
        """
		# _, labels = cv2.connectedComponents(maskImg)
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(maskImg, connectivity=8)
		cv2.imwrite(os.path.join(self.outpath, "cell_mask.tif"), labels)
		"""
        Merge 表达矩阵与细胞label
        """

		tissuedf = pd.DataFrame()
		dst = np.nonzero(labels)

		tissuedf['x'] = dst[1]
		tissuedf['y'] = dst[0]
		tissuedf['label'] = labels[dst]

		celldf = pd.merge(self.genedf, tissuedf, on=['x', 'y'], how='inner')
		with gzip.open(os.path.join(self.outpath, "{0}.cell.gem.gz".format(self.snId)), 'wt') as writer:
			writer.write(self.info)
			celldf.to_csv(writer, sep='\t', index=False)

		statDf = pd.DataFrame()
		statDf['label'] = range(1, stats.shape[0])
		statDf['x'] = stats[..., 1][1:]
		statDf['y'] = stats[..., 0][1:]
		statDf['CellArea'] = stats[..., 4][1:]
		statDf.to_csv(os.path.join(self.outpath, "cell_stat.txt"), sep="\t", index=False)

		t1 = time.time()
		print("Dumping cell segmentation result done. time used: {:.2f}".format(t1 - t0))
		"""
        画cell bin统计图，并将cell bin的统计结果输出到log文件中
        """
		celldf['Gene Number'] = celldf['geneID']
		celldf['MID Count'] = celldf['MIDCount']
		celldf['DNB Number'] = celldf['x'].astype(str) + "-" + celldf['y'].astype(str)
		groupdf = celldf.groupby(['label'])
		sum_info = groupdf[['MID Count']].sum()
		count_info = groupdf[['Gene Number', 'DNB Number']].nunique()
		statdf = pd.concat([sum_info, count_info], axis=1)

		readsdf = pd.merge(celldf[['x', 'y', 'label']], dnbdf, how='inner')
		groupreadsdf = readsdf['reads'].groupby(readsdf['label']).sum()
		mReads = round(groupreadsdf.values.mean(), 2)
		medReads = np.median(groupreadsdf.values)
		logpath = os.path.join(self.outpath, "TissueCut.log")
		log = open(logpath, "a")
		self.gemStat(statdf, mReads, medReads, log, 0)
		log.close()

	def read_Geojson(self, geoFile):
		"""
        读取geojson，并转化为bit8的mask
        """
		import geojson
		mask = np.zeros(self.ori_shape, np.uint8)
		with open(geoFile, "r") as geofile:
			gj = geojson.load(geofile)

		for i in gj['geometries']:
			cv2.fillPoly(mask, np.array(i["coordinates"]), 255)

		return mask

	def TissueStat(self, mergedf, dnbdf, conArea, num_dnb, dnbreads, maxbinSize=200, step=50):
		"""
        结果统计
        mergedf: 组织区域表达矩阵
        coor：组织区域dnb坐标
        dnbdf：组织区域 x, y, reads 对应表格
        conArea：组织区域面积（dnb）
        num_dnb: 组织下dnb数量
        dnbreads: 总的reads数
        maxbinSize/step: 输出某个binSize下的统计结果，default 200/50 即从1开始，输出步长为50的binning结果 （1，50，100，150，200）

        统计结果包括基因数，UMI数，reads数等 以及对应的小提琴图/散点图
        """

		num_gene = mergedf['geneID'].nunique()
		num_umi = mergedf['MIDCount'].sum()
		umiFraction = num_umi / self.total_umi * 100
		print("Gene type: ", num_gene)
		print("Total umi: ", num_umi)

		# mergetis = pd.merge(coor, dnbdf, how='inner')

		### fraction reads in spots under tissue
		# dnbreads = dnbdf['reads'].sum()
		print("DNB reads: ", dnbreads)
		tisreads = dnbdf['reads'].sum()
		print("Tissue area reads: ", tisreads)
		if dnbreads != 0:
			fraction = tisreads / dnbreads * 100
		else:
			fraction = 0
		print("The fraction: ", fraction)
		logpath = os.path.join(self.outpath, "TissueCut.log")
		log = open(logpath, "w")
		if self.stainUsed:
			log.write("# Tissue Statistic Analysis with Stain Image\n")
		else:
			log.write("# Tissue Statistic Analysis\n")
		log.write("Contour_Area\t{}\nNumber_of_DNB_Under_Tissue\t{}\nRatio\t{:.2f}% \n".format(conArea, num_dnb, (
					num_dnb / conArea) * 100))
		log.write("Total_Gene_Type\t{}\n".format(num_gene))
		log.write("MID_Counts\t{}\n".format(num_umi))
		log.write("Fraction_MID_in_Spots_Under_Tissue\t{:.2f}%\n".format(umiFraction))
		log.write("Reads_Under_Tissue\t{}\n".format(tisreads))
		log.write("Fraction_Reads_in_Spots_Under_Tissue\t{:.2f}%\n".format(fraction))

		### Mean reads per spots/ median genes/ median UMI
		for b in range(0, maxbinSize + 1, step):
			if b == 0:
				b = 1

			df = self.binStat(mergedf, b)
			_, meanReads, medReads = self._binStat(dnbdf, b, "reads")
			self.gemStat(df, meanReads, medReads, log, b)

		log.close()

	def binStat(self, mergedf, binSize):
		bindf = pd.DataFrame()
		bindf['x'] = mergedf['x'] // binSize
		bindf['y'] = mergedf['y'] // binSize
		# bindf['Read Count'] = mergedf['reads']
		bindf['Gene Number'] = mergedf['geneID']
		bindf['MID Count'] = mergedf['MIDCount']
		bindf['DNB Number'] = mergedf['x'].astype(str) + "-" + mergedf['y'].astype(str)

		groupdf = bindf.groupby(['x', 'y'])
		sum_info = groupdf[['MID Count']].sum()
		count_info = groupdf[['Gene Number', 'DNB Number']].nunique()
		statdf = pd.concat([sum_info, count_info], axis=1)

		return statdf

	def _binStat(self, mergedf, binSize, t):
		bindf = pd.DataFrame()
		bindf['x'] = mergedf['x'] // binSize
		bindf['y'] = mergedf['y'] // binSize
		bindf[t] = mergedf[t]

		if t == 'geneID':
			if binSize == 1:
				groupdf = mergedf["geneID"].groupby([mergedf['x'], mergedf['y']]).count()
			else:
				groupdf = bindf[t].groupby([bindf['x'], bindf['y']]).nunique()
		else:
			groupdf = bindf[t].groupby([bindf['x'], bindf['y']]).sum()
		if bindf.empty:
			mean = 0
			med = 0
		else:
			mean = round(groupdf.values.mean(), 2)
			med = np.median(groupdf.values)
		return groupdf, mean, med

	def bin1GeneStat(self, mergedf):
		groupdf = mergedf["geneID"].groupby([mergedf['x'], mergedf['y']]).count()
		mean = round(groupdf.values.mean(), 3)
		med = np.median(groupdf.values)
		return groupdf, mean, med

	def gemStat(self, df, mReads, medReads, log, b):
		if (b != 1):
			scapath = os.path.join(self.figpath, "scatter_{0}x{0}_MID_gene_counts.png".format(b if b != 0 else "cell"))
			violinpath = os.path.join(self.figpath, "violin_{0}x{0}_MID_gene.png".format(b if b != 0 else "cell"))
			statisticPath = os.path.join(self.figpath,
			                             "statistic_{0}x{0}_MID_gene_DNB.png".format(b if b != 0 else "cell"))
			plt.figure(figsize=(5, 5))
			# sns.scatterplot(x=df['n_counts'], y=df['n_genes'], edgecolor="gray", color="gray")
			plt.scatter(df['MID Count'], df['Gene Number'], color="gray", edgecolors="gray", s=0.8)
			plt.grid()
			plt.xlabel("MID Count")
			plt.ylabel("Gene Number")
			plt.savefig(scapath, format="png", bbox_inches="tight")

			plt.figure(figsize=(10, 6))
			plt.subplot(121)
			sns.violinplot(y=df['MID Count'])
			sns.stripplot(y=df['MID Count'], jitter=0.4, color="black", size=0.8)
			plt.ylabel("")
			plt.title("MID Count")
			plt.subplot(122)
			sns.violinplot(y=df['Gene Number'])
			sns.stripplot(y=df['Gene Number'], jitter=0.4, color="black", size=0.8)
			plt.ylabel("")
			plt.title("Gene Number")
			plt.savefig(violinpath, format="png", bbox_inches="tight")

			if self.develop:
				g = sns.FacetGrid(pd.melt(df[['MID Count', 'Gene Number', 'DNB Number']]), col='variable',
				                  hue='variable', sharex=False, sharey=False, height=8, palette='Set1')
				g = (g.map(sns.distplot, "value", hist=False, rug=True))
				plt.savefig(statisticPath)

		meanStat = df.mean()
		medStat = df.median()

		mGene, mUMI = round(meanStat['Gene Number'], 3), round(meanStat['MID Count'], 3)
		medGene, medUMI = round(medStat['Gene Number'], 3), round(medStat['MID Count'], 3)
		print("binSize={0}".format(b if b != 0 else "cell"))
		print("mean/median reads: ", mReads, medReads)
		print("mean/median gene type: ", mGene, medGene)
		print("mean/median UMI: ", mUMI, medUMI)
		log.write("\nBin_Size\t{0}\n".format(b if b != 0 else "cell"))
		log.write("Mean_Reads_per_Spot\t{}\nMedian_Reads_per_Spot\t{}\n".format(mReads, medReads))
		log.write("Mean_Gene_Type_per_Spot\t{}\nMedian_Gene_Type_per_Spot\t{}\n".format(mGene, medGene))
		log.write("Mean_MID_per_Spot\t{}\nMedian_MID_per_Spot\t{}\n".format(mUMI, medUMI))

	def drawBin100imgAndcompleteGef(self):
		bin100img=self.CreatImgFromFile(self.binSize,normalize=True)
		dnbMattingImgFile = os.path.join(self.outpath, "{0}X{0}_contour_image.png".format(self.binSize))
		cv2.imwrite(dnbMattingImgFile, bin100img)
		del bin100img
		gc.collect()
		filterGenefile = os.path.join(self.outpath, "{0}.gef".format(self.snId))
		# glog.info("start to generate sn.gef")
		# generate bgef by shell, because cpython fail to control the memory usage
		# generate_bgef(self.infile, filterGenefile, stromics=self.omics)

		# glog.info("generate sn.gef completed")
		self.parseGefOffset(filterGenefile)
		temppath = os.path.join(self.outpath, "dnb_merge")
		os.makedirs(temppath, exist_ok=True)
		gefplot.save_exp_heat_map(filterGenefile, os.path.join(temppath, "bin200.png"))
		gc.collect()
		pass
	def CreatImgFromFile(self,binSize,normalize=False):

		f = h5py.File(self.infile, 'r')

		# f = h5py.File('/Users/berry/tmp/SS200000003BR_B3.tissue.gef', 'r')
		self.rawGefMinX = f['/geneExp/bin1/expression'].attrs['minX'][0]
		self.rawGefMinY = f['/geneExp/bin1/expression'].attrs['minY'][0]
		self.rawGefMaxX = f['/geneExp/bin1/expression'].attrs['maxX'][0]
		self.rawGefMaxY = f['/geneExp/bin1/expression'].attrs['maxY'][0]
		# cols = self.rawGefMinX + self.rawGefMaxX
		cols = self.rawGefMaxX-self.rawGefMinX
		# rows = self.rawGefMinY + self.rawGefMaxY
		rows = self.rawGefMaxY-self.rawGefMinY
		cols = cols // binSize + 1
		rows = rows // binSize + 1
		if(binSize>1):
			binImg = np.zeros([rows, cols], dtype=np.uint16)
		else:
			binImg=np.zeros([rows, cols], dtype=np.uint16)
		exp = f['/geneExp/bin1/expression']

		self.resolution = exp.attrs['resolution']
		self.version = f.attrs['version']
		self.tissueOffsetX = 0
		self.tissueOffsetY = 0
		totalNum = exp.shape[0]
		batch = totalNum // 4
		xmin = np.iinfo(np.uint32).max
		xmax = 0
		ymin = np.iinfo(np.uint32).max
		ymax = 0
		print("bin100 img shape:", binImg.shape)
		for i in range(0, totalNum, batch):
			endIdx = totalNum if i + batch > totalNum else i + batch
			expNp = np.array(exp.fields(['x', 'y', 'count'])[i:endIdx])
			# expNp['x'] = (expNp['x'] + self.rawGefMinX) // binSize
			expNp['x'] = expNp['x'] // binSize
			# expNp['y'] = (expNp['y'] + self.rawGefMinY) // binSize
			expNp['y'] = expNp['y'] // binSize
			# txmax = expNp['x'].max()
			# txmin = expNp['x'].min()
			# tymax = expNp['y'].max()
			# tymin = expNp['y'].min()
			# if txmax > xmax:
			# 	xmax = txmax
			# if txmin < xmin:
			# 	xmin = txmin
			# if tymax > ymax:
			# 	ymax = tymax
			# if tymin < ymin:
			# 	ymin = tymin
			# if(binSize>1):
			# 	self.total_umi+=expNp['count'].sum()
			for ele in expNp:
				binImg[ele[1]][ele[0]] += ele[2]
			glog.info("processed:"+str(i // batch))
			del expNp
			gc.collect()
		f.close()
		print("img shape:", binImg.shape)
		# binImg=binImg[0:ymax-ymin+1,0:xmax-xmin+1]
		gc.collect()
		print("img shape:", binImg.shape)
		print("bin size: ", binSize)
		# print("Size of bin image: ", xmin, xmax, ymin, ymax)
		f.close()
		if not normalize:
			return binImg
		else:
			Imin, Imax = binImg.min(), binImg.max()
			Omin, Omax = 0, 255
			a = float(Omax - Omin) / (Imax - Imin)
			b = Omin - a * Imin
			img = a * binImg + b
			img = img.astype(np.uint8)
			del binImg
			# del img
			gc.collect()
			return img
