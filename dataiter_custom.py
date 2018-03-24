import numpy as np
import random
#import matplotlib.pyplot as plt
import cv2
import mxnet as mx
import os
import math
from mxnet.io import DataIter, DataBatch


class FileIter(DataIter):
	def __init__(self, data_shapes, set_num, per_set_num, duplicate_num, ctx, data_name="data", label_name="label"):
		#duplicate_num <= set_num/2
		super(FileIter, self).__init__()
		self.batch_size = set_num*per_set_num
		self.data_shapes = data_shapes
		self.set_num = set_num
		self.per_set_num = per_set_num
		self.duplicate_num = duplicate_num
		self.data_name = data_name
		self.label_name = label_name
		self.ctx = ctx
		#self.data = mx.nd.zeros((self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]), self.ctx)
		#self.label = mx.nd.zeros((self.batch_size, ), self.ctx)
		
		self.train = mx.io.ImageRecordIter(
			path_imgrec = '../hangzhougongan_train.rec',
			data_shape  = self.data_shapes,
			batch_size  = self.batch_size,
			#shuffle     = 0,                 
			round_batch = 0,
			#scale       = 0.007843137,
			#mean_r      = 127.5,
			#preprocess_threads = 2,
		)

	def _shuffle(self):
		random.shuffle(self.train_list)

	@property
	def provide_data(self):
		return [(self.data_name, (self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]))]

	@property
	def provide_label(self):
		return [(self.label_name, (self.batch_size, ))]

	def get_total_list(self):
		self.train.reset()

		cur_num = 0
		label_img = {} 

		while self.train.iter_next():
			cur_num += 1 
			img_batch = self.train.getdata()[0].asnumpy()
			label_batch = self.train.getlabel().asnumpy()
			for i in range(len(label_batch)):
				label = int(label_batch[i])
				img = img_batch[i].astype(np.uint8)
				label_img[str(label)+' '+str(cur_num)+' '+str(i)] = img
			#if cur_num == 10:
			#	break

		self.sorted_label_img = sorted(label_img.items(), key = lambda x:int(x[0].split(' ')[0])) #label = sorted_label_img[i][0]   img = sorted_label_img[i][1] 

	def reset(self):
		self.get_total_list()
		#tmp_list[i] tuple -> list
		self.tmp_list = []
		for i in range(len(self.sorted_label_img)):
			self.tmp_list.append([self.sorted_label_img[i][0], self.sorted_label_img[i][1]])
			self.tmp_list[i].append(1)

	def iter_next(self):
		if self.get_label_idx():
			return len(self.label_idx) >= self.set_num
		else:
			return False

	def get_label_idx(self):
		self.label_idx = {}
		if self.tmp_list:
			tmp = int(self.tmp_list[0][0].split(' ')[0])  #the first one
			self.label_idx[tmp] = 0
			for i in range(len(self.tmp_list)):
				cur_label = int(self.tmp_list[i][0].split(' ')[0])
				if tmp != cur_label:
					tmp = cur_label
					self.label_idx[cur_label] = i
			return True
		else:
			return False

	def get_train_list(self):
		random_list = random.sample(self.label_idx.keys(), self.set_num)  #label list
		self.duplicate_clean_position = []
		self.duplicate_noise_position = []
		self.others_position = []
		if self.duplicate_num==0:
			random_list = random_list
		else:
			duplicate_list = random.sample(random_list, self.duplicate_num)
			duplicate_tmp = 0
			for i in range(len(random_list)):
				if random_list[i] in duplicate_list:
					for j in range(self.per_set_num):
						self.duplicate_clean_position.append(i*self.per_set_num+j)
					#continue
				else:
					if duplicate_tmp < len(duplicate_list):
						random_list[i] = duplicate_list[duplicate_tmp]
						for j in range(self.per_set_num):
							self.duplicate_noise_position.append(i*self.per_set_num+j)
						duplicate_tmp += 1
					elif duplicate_tmp == len(duplicate_list):
						for j in range(self.per_set_num):
							self.others_position.append(i*self.per_set_num+j)

		total_list = []
		# len(self.label_idx) = 53950
		for i in range(self.set_num):
			if random_list[i] == max(self.label_idx.keys()):  #the last one
				per_random_list = [random.randint(self.label_idx[random_list[i]], len(self.tmp_list)-1) for _ in range(self.per_set_num)]
			else:
				k = 0
				while not(random_list[i]+k+1 in self.label_idx.keys()):
					k += 1
				per_random_list = [random.randint(self.label_idx[random_list[i]], self.label_idx[random_list[i]+k+1]-1) for _ in range(self.per_set_num)] 
			total_list.extend(per_random_list)

		self.train_list = total_list
		import pdb; pdb.set_trace()

	def next(self):
		if self.iter_next():
			self.get_train_list()
			data = np.ones((self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]), dtype='float32')
			label = np.zeros((self.batch_size, ), dtype=np.float32)
			#self._shuffle()
			
			if self.duplicate_num==0:
				for i in range(len(self.train_list)):
					prob = np.random.uniform()
					if prob <= 0.30:
						data_origin = self.tmp_list[self.train_list[i]][1].astype('float32').transpose((1, 2, 0))
						data_BGR = cv2.cvtColor(data_origin, cv2.COLOR_RGB2BGR)
						kernel, anchor = motion_blur(random.randint(20, 60), random.randint(20, 60))
						data_BGR = cv2.filter2D(data_BGR, -1, kernel, anchor=anchor)
						data[i] = cv2.cvtColor(data_BGR, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
					else:
						data[i] = self.tmp_list[self.train_list[i]][1].astype('float32')

			else:
				for i in range(len(self.train_list)):
					if i in self.duplicate_clean_position:
						data[i] = self.tmp_list[self.train_list[i]][1].astype('float32')
					elif i in self.duplicate_noise_position:
						prob = np.random.uniform()
						if prob <= 0.80:
							data_origin = self.tmp_list[self.train_list[i]][1].astype('float32').transpose((1, 2, 0))
							data_BGR = cv2.cvtColor(data_origin, cv2.COLOR_RGB2BGR)
							kernel, anchor = motion_blur(random.randint(20, 60), random.randint(20, 60))
							data_BGR = cv2.filter2D(data_BGR, -1, kernel, anchor=anchor)
							data[i] = cv2.cvtColor(data_BGR, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
						else:
							data[i] = self.tmp_list[self.train_list[i]][1].astype('float32')
					elif i in self.others_position:
						prob = np.random.uniform()
						if prob <= 0.30:
							data_origin = self.tmp_list[self.train_list[i]][1].astype('float32').transpose((1, 2, 0))
							data_BGR = cv2.cvtColor(data_origin, cv2.COLOR_RGB2BGR)
							kernel, anchor = motion_blur(random.randint(20, 60), random.randint(20, 60))
							data_BGR = cv2.filter2D(data_BGR, -1, kernel, anchor=anchor)
							data[i] = cv2.cvtColor(data_BGR, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
						else:
							data[i] = self.tmp_list[self.train_list[i]][1].astype('float32')

			occlusion_aug(self.batch_size, self.data_shapes, max_w=60, max_h=100, 
							min_prob=0.0, max_prob=0.3, img=data)
			
			'''
			for i in range(len(data)):
				cv2.imwrite(str(i)+'.jpg', cv2.cvtColor(data[i].transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))
				#print i
			import pdb; pdb.set_trace()
			'''

			for i in range(self.batch_size):
				data_aug = data[i]
				data_mean = data_aug
				data_mean[0,:] = data_aug[0,:]-127.5
				data_scale = data_mean/127.5

				data[i] = data_scale
				label[i] = int(self.tmp_list[self.train_list[i]][0].split(' ')[0])

			self.data = [mx.nd.array(data, self.ctx)]
			self.label = [mx.nd.array(label, self.ctx)]
			#import pdb; pdb.set_trace()
			self.train_list = list(set(self.train_list))

			#delete samples which are trained already
			del_num = 0
			for i in range(len(self.train_list)):
				self.tmp_list[self.train_list[i]][2] = 0
			for i in range(len(self.tmp_list)):
				if self.tmp_list[i-del_num][2] == 0:
					del self.tmp_list[i-del_num]
					del_num += 1

			return DataBatch(data=self.data, label=self.label)
		else:
			raise StopIteration


def occlusion_aug(batch_size, img_shape, max_w, max_h, min_prob, max_prob, img): 
	shape = [batch_size] + list(img_shape)
	channel_num = img_shape[1]
	img_w = shape[3]
	img_h = shape[2]
	prob = np.random.uniform(min_prob, max_prob)
	rand_num = int(prob * batch_size)
	if rand_num <= 0:
		return img
	rand_index = np.random.choice(batch_size, rand_num, False)
	rand_source = np.random.randint(0, 1000000, rand_num * 4)
	x_rand = rand_source[0:rand_num] % img_w #np.random.randint(0, img_w, rand_num)
	y_rand = rand_source[rand_num:2*rand_num] % img_h  #np.random.randint(0, img_h, rand_num)
	w_rand = rand_source[rand_num*2:3*rand_num] % max_w + 1 #np.random.randint(0, img_w, rand_num)
	h_rand = rand_source[rand_num*3:4*rand_num] % max_h + 1#np.random.randint(0, img_h, rand_num)

	indices = np.where(img_w - (x_rand + w_rand) < 0)
	w_rand[indices] = img_w - x_rand[indices]
	indices = np.where(img_h - (y_rand + h_rand) < 0)
	h_rand[indices] = img_h - y_rand[indices]
	for k in range(rand_num):
		index = rand_index[k]
		img[index][:,y_rand[k]:y_rand[k] + h_rand[k],x_rand[k]:x_rand[k]+w_rand[k]] = random.sample([0, 255], 1)[0]
	return img

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
