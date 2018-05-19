import cv2
import mxnet as mx
import numpy as np
import random
import math

def motion_blur(length, angle):
    half = length / 2
    EPS = np.finfo(float).eps
    alpha = (angle - math.floor(angle / 180) * 180) /180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1

    # blur kernel size
    sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    psf1 = np.zeros((sy, sx))

    # psf1 is getting small when (x,y) move from left-top to right-bottom
    # at this moment (x,y) is moving from right-bottom to left-top
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i*i + j*j)
            if  rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
            if psf1[i][j] < 0:
                psf1[i][j] = 0

    # anchor is (0,0) when (x,y) is moving towards left-top
    anchor = (0, 0)
    # anchor is (width, heigth) when (x, y) is moving towards right-top
    # flip kernel at this moment
    if angle < 90 and angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif angle > -90 and angle < 0: # moving towards right-bottom
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
    elif anchor < -90: # moving towards left-bottom
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1 / psf1.sum()
    return psf1, anchor


if __name__ == '__main__':
	data_iter = mx.io.ImageRecordIter(
	    path_imgrec="2rec_version/hangzhougongan_2rec_rect_val.rec", # the target record file
	    data_shape=(3, 72, 72), # output data shape. An 227x227 region will be cropped from the original image.
	    batch_size=200, # number of samples per batch
	    #resize=256 # resize the shorter edge to 256 before cropping
	    # ... you can add more augumentation options as defined in ImageRecordIter.
	    )

	data_iter.reset()
	batch = data_iter.next()
	data = batch.data[0].asnumpy()
	for i in range(len(data)):
		prob = np.random.uniform()
		if prob <= 0.80:
			data_origin = data[i].astype('float32').transpose((1, 2, 0))
			kernel, anchor = motion_blur(random.randint(20, 60), random.randint(20, 60))
			#data[i] = cv2.filter2D(data_origin, -1, kernel, anchor=anchor).transpose((2, 0, 1))

			image = cv2.cvtColor(cv2.filter2D(data_origin, -1, kernel, anchor=anchor), cv2.COLOR_RGB2BGR)
			cv2.imwrite('test_img_for_2rec/'+str(i)+'.jpg', image)
		else:
			data_origin = data[i].astype('float32').transpose((1, 2, 0))
			image = cv2.cvtColor(data_origin, cv2.COLOR_RGB2BGR)
			cv2.imwrite('test_img_for_2rec/'+str(i)+'.jpg', image)


