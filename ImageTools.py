import pandas as pd
import numpy as np
from scipy import sparse

def CreatImg(df, binSize, normalize=True):
    """
    根据表达矩阵生成图片
    """

    bindf = pd.DataFrame()
    bindf['x'] = df['x'] // binSize
    bindf['y'] = df['y'] // binSize
    if 'values' in df.columns:
        bindf['MIDCount'] = df['values']
    elif "UMICount" in df.columns:
        bindf['MIDCount'] = df['UMICount']
    else:
        bindf['MIDCount'] = df['MIDCount']
    bindf = bindf['MIDCount'].groupby([bindf['x'], bindf['y']]).sum().reset_index()
    xmin, xmax = bindf['x'].min(), bindf['x'].max()
    ymin, ymax = bindf['y'].min(), bindf['y'].max()
    print("bin size: ", binSize)
    print("Size of bin image: ", xmin, xmax, ymin, ymax)
    bindf['x'] = bindf['x'] - xmin
    bindf['y'] = bindf['y'] - ymin
    sparseMt = sparse.csr_matrix((bindf['MIDCount'].astype(np.uint16), (bindf['y'], bindf['x'])))
    image = sparseMt.toarray()
    print("bin image shape ", image.shape)
    if not normalize:
        return image
    ### Create image
    else:
        Imin, Imax = image.min(), image.max()
        Omin, Omax = 0, 255
        a = float(Omax-Omin)/(Imax-Imin)
        b = Omin - a*Imin
        img = a*image + b
        img = img.astype(np.uint8)
        return img

def __CreatOriImg(genedf):
    ## deprecated
    x1, x2 = genedf['x'].min(), genedf['x'].max()
    y1, y2 = genedf['y'].min(), genedf['y'].max()
    print("size of original image: ", x1, x2, y1, y2)
    shape = (y2 - y1 + 1, x2 - x1 + 1)
    oriimage = np.zeros(shape, np.uint8)
    print("original image shape ", oriimage.shape)
    ### Create image
    for i in range(len(genedf)):
        y = genedf['y'].iloc[i] - y1
        x = genedf['x'].iloc[i] - x1
        oriimage[y][x] = 255

    oriimage = oriimage.astype(np.uint8)
    print("Create original image done. ")
    return oriimage

def readDNB(DNBfile, offsetX, offsetY):
    """
    read dnbfile
    """
    
    tmpdf = pd.read_csv(DNBfile, sep='\t', names=["x", "y", "reads"], dtype=np.uint32)
    tmpdf.loc[(tmpdf['x']>=offsetX)&(tmpdf['y']>=offsetY)]
    tmpdf['x'] = tmpdf['x'] - offsetX
    tmpdf['y'] = tmpdf['y'] - offsetY
    return tmpdf    

def __readDNB(DNBfile):
    ## deprecated
    tmpdf = pd.read_csv(DNBfile, sep='\t')
    col = tmpdf.columns
    dnbdf = pd.DataFrame()
    dnbdf['x'] = [col[0]] + tmpdf[col[0]].tolist()
    dnbdf['y'] = [col[1]] + tmpdf[col[1]].tolist()
    dnbdf['reads'] = [col[2]] + tmpdf[col[2]].tolist()
    dnbdf = dnbdf.astype(np.uint32)

    groupdf = dnbdf['reads'].groupby([dnbdf['x'], dnbdf['y']]).sum()

    result = pd.DataFrame()
    result['x'] = [i[0] for i in groupdf.index]
    result['y'] = [i[1] for i in groupdf.index]
    result['reads'] = groupdf.values
    return result

def Imgsplit(binImg, n):
    step_0 = binImg.shape[0] // n
    step_1 = binImg.shape[1] // n
    img_data = []
    
    for i in range(n):
        for j in range(n):
            img = binImg[i*step_0 : (i+1)*step_0, j*step_1:(j+1)*step_1]
            img_data.append(img)
    return img_data

def Imgcombine(img_data, n):
    mask = []
    for i in range(n):
        tmp = np.concatenate(img_data[i*n:(i+1)*n], axis=1)
        mask.append(tmp)
    Imgmask = np.concatenate(mask, axis=0)
    return Imgmask
