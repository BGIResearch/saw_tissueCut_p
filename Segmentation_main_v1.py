import os, sys
from optparse import OptionParser

from TissueSegmentation import tissueSegmentation
from CellSegmentation import cellSegmentation

import numexpr
import time

"""
input: merge_GetExp_gene.txt

return: 
tissue:
merge_GetExp_gene.txt (cutted) | Tissuefig (scatter/violin plot binSize50-200) | tissueSegmentation.log

cell:
merge_GetExp_gene.txt (labeled) | cellSegmentation.log | Cellfig (stat figs)
"""

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="infile", help="The input gene expression file. ")
    parser.add_option("-o", "--outpath", dest="out_path", help="The output path. ")
    parser.add_option('-t', "--tissue", dest="tissue", default="tissue", help="The tissue type. ")
    parser.add_option('--thread', dest="thread", default=4, help="Number of threads used in numpy.")
    parser.add_option('-f', dest="f", default=1, type=int, help="Image flip code. 1 for flip vertically, 0 for flip horizontally.")
    parser.add_option('-m', "--multi", dest="multi", default=5, type=int, help="Resize image.")
    parser.add_option("--snId", dest="snId", default="stereo-chip", help="slide number to identify the stereo-chip.")
    parser.add_option("--omics",dest="omics",default="Transcriptomics",help="omics information which will be write to gef")
    ### Tissue segmentation parameters
    parser.add_option("--dnbfile", dest='dnbfile', help="The barcode reads count file. ")
    # parser.add_option("--action", dest='action', default=0, type=int, help="0: for TissueCut. 1: for Registered data. ")
    parser.add_option('-s', '--stain', dest='stainfile', help="The directory contain staining image for registration. If action == 1, this parameter must be sepecified. ")
    parser.add_option("--bin", dest="bin", type=int, default=100, help="The binSize used to creat tiff image.default 100.")
    # parser.add_option("--binSizeNoStain", dest="binSizeNoStain", type=int, default=1, help="The binSize used to do deep learning when no stain file.")
    parser.add_option("--shape", dest="shape", type=int, default=3, help="The shape of the amplifier filter. default 3")
    parser.add_option("--ampfactor", dest="ampfactor", type=int, default=1, help="The amplify factor. default 1")
    parser.add_option("--meanValue", dest="meanValue", type=int, default=12, help="The min mean threshold. ")
    parser.add_option("--platform", dest="platform", default='T1', help="Sequencing platform. ")
    parser.add_option("--kernelSize", dest="kernel", type=int, default=13, help="The kernel size of blur filter. default 10")
    parser.add_option("--low_thresh", dest='low_thresh', type=int, default=0, help="Threshold. default 0.")
    parser.add_option("--high_thresh", dest="high_thresh", type=int, default=255, help="threshold. default 255.")

    ### Cell segmentation parameters
    parser.add_option("--binSize", dest="binSize", default=5, type=int, help="bin size")
    parser.add_option("--minArea", dest="minArea", default=200, type=int, help="min Cell area. default 200 ")
    parser.add_option("--maxArea", dest="maxArea", default=50000, type=int, help="max Cell area. default 50,000")

    ### Develop version
    parser.add_option("-d", "--develop", dest="develop", action="store_true")
    parser.set_defaults(develop=False)
    
    opts, args = parser.parse_args()


    if opts.infile == None or opts.out_path == None:
        sys.exit(not parser.print_help())
    
    # if opts.tissue is None or opts.dnbfile == None:
    #     sys.exit(not parser.print_help())

    outpath = opts.out_path
    os.makedirs(outpath, exist_ok=True)

    numexpr.set_num_threads(opts.thread)

    if opts.tissue.upper() == 'CELL':
        runCellsegmentation(opts.infile, opts.dnbfile, outpath, opts.binSize, opts.snId, opts.minArea, opts.maxArea)
    else:
        runTissuesegmentation(opts.infile, opts.dnbfile, opts.stainfile, opts.snId, opts.f, opts.multi, outpath, opts.bin, opts.shape, opts.ampfactor, opts.meanValue, opts.platform, opts.kernel, opts.low_thresh, opts.high_thresh, opts.develop,opts.omics)

def runTissuesegmentation(infile, dnbfile, stainfile, snId, flip_code, multi, outpath, binSize, amp_shape, amp_factor, min_mean, platform, struct_kernel, low_thresh, high_thresh, develop,omics):
    """
    组织抠图
    输入：
    dnbfile：每个dnb对应捕获到的raw reads数（可能之后不需要统计这部分，这部分输入可以删除）format: x, y, reads
    stainfile：配准图，如为None则运行自动扣图函数
    outpath：输出路径
    binSize：处理图片时的binSize
    amp_shape/amp_factor: 增强图片时的kernel
    min_mean：表达量的阈值，如果低于该阈值则对图像进行增强处理
    platform: 测序平台

    输出：
    组织区域的表达矩阵
    统计结果log
    基因数/UMI数 小提琴图/散点图
    """
    print("reading data..")
    t0 = time.time()
    tissueSeg = tissueSegmentation(infile, outpath, binSize, snId, develop,omics)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\tstart tissueSeq.process')
    tissueSeg.process(stainfile, dnbfile, flip_code, multi, amp_shape, amp_factor, min_mean, platform, struct_kernel, low_thresh, high_thresh)
    
    
def runCellsegmentation(infile, dnbfile, outpath, binSize, snId, minArea, maxArea):
    cellSeg = cellSegmentation(infile, dnbfile, outpath, binSize, snId, minArea, maxArea)
    cellSeg.process()
        
if __name__ == '__main__':
    main()
