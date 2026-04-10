import cv2
import numpy as np
import os
import tempfile

def recompress_diff(imorig, checkDisplacements):
    minQ = 51
    maxQ = 100
    stepQ = 1

    if checkDisplacements == 1:
        maxDisp = 7
    else:
        maxDisp = 0

    mins = []
    Output = []

    smoothing_b = 17
    Offset = (smoothing_b - 1) // 2

    height, width, _ = imorig.shape

    dispImages = []

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_resave_path = tmp_file.name

    try:
        for ii in range(minQ, maxQ + 1, stepQ):
            cv2.imwrite(tmp_resave_path, imorig, [int(cv2.IMWRITE_JPEG_QUALITY), ii])
            tmpResave = cv2.imread(tmp_resave_path).astype(float)
            Deltas = []
            overallDelta = []

            for dispx in range(maxDisp + 1):
                for dispy in range(maxDisp + 1):
                    DisplacementIndex = dispx * 8 + dispy + 1
                    tmpResave_disp = tmpResave[dispx:, dispy:, :]
                    imorig_disp = imorig[:height-dispx, :width-dispy, :].astype(float)
                    Comparison = np.square(imorig_disp - tmpResave_disp)

                    h = np.ones((smoothing_b, smoothing_b)) / smoothing_b**2
                    Comparison = cv2.filter2D(Comparison, -1, h)

                    Comparison = Comparison[Offset:-Offset, Offset:-Offset, :]
                    Deltas.append(np.mean(Comparison, axis=2))
                    overallDelta.append(np.mean(Deltas[DisplacementIndex - 1]))

            minOverallDelta, minInd = min(overallDelta), np.argmin(overallDelta)
            mins.append(minInd)
            Output.append(minOverallDelta)
            delta = Deltas[minInd]
            delta_range = np.max(delta) - np.min(delta)
            delta = (delta - np.min(delta)) / delta_range if delta_range > 0 else delta * 0

            dispImages.append(cv2.resize(delta.astype(np.float32), (delta.shape[1] // 4, delta.shape[0] // 4), interpolation=cv2.INTER_LINEAR))
    finally:
        try:
            os.unlink(tmp_resave_path)
        except OSError:
            pass

    OutputY = Output
    OutputX = list(range(minQ, maxQ + 1, stepQ))
    _xmax, _imax, _xmin, imin = cv2.minMaxLoc(np.array(OutputY))
    imin = sorted(imin)
    Qualities = [i * stepQ + minQ - 1 for i in imin]

    return OutputX, OutputY, dispImages, imin, Qualities, mins

def clean_up_image(filename):
    im = cv2.imread(filename)

    dots = filename.rfind('.')
    extension = filename[dots:]
    
    if extension.lower() == '.gif' and im.shape[2] < 3:
        im_gif, gif_map = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im_gif = im_gif[:, :, 0]
        im = np.uint8(cv2.cvtColor(im_gif, cv2.COLOR_GRAY2RGB) * 255)

    if im.shape[2] < 3:
        im[:, :, 1] = im[:, :, 0]
        im[:, :, 2] = im[:, :, 0]

    if im.shape[2] > 3:
        im = im[:, :, 0:3]

    if im.dtype == np.uint16:
        im = np.uint8(np.floor(im / 256))

    return im

def img_heatmap_cd(impath):
    im = clean_up_image(impath)
    checkDisplacements = 0
    OutputX, _OutputY, dispImages, _imin, _Qualities, _mins = recompress_diff(im, checkDisplacements)
    return dispImages, OutputX
