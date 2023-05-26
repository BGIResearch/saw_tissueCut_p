PREFIX=T194_FP200000340BR_C4

OUT_PATH=/Project/MouseBrain/$PREFIX/segmentation
STAIN_IMG=/Project/MouseBrain/$PREFIX/2021*.tif
HE=/Project/MouseBrain/$PREFIX/segmentation/TissueFig/HE*.png

python3 /Project/MouseBrain/ImageSegmentation/ImagePyramid.py \
-s $STAIN_IMG \
-o $OUT_PATH \
-i $HE

echo $OUT_PATH/*.ssDNA.h5
