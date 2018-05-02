import numpy as np
import random
#import matplotlib.pyplot as plt
import cv2
import mxnet as mx
import os
import math
from mxnet.io import DataIter, DataBatch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FileIter(DataIter):
	def __init__(self, data_shapes, set_num, per_set_num, duplicate_num, hdfs_path, max_num_of_duplicate_set,\
				rec_name, max_times_epoch, data_type, data_name="data", label_name="label"):
		#duplicate_num <= set_num/2
		super(FileIter, self).__init__()
		self.batch_size = set_num*per_set_num
		self.data_shapes = data_shapes
		self.set_num = set_num
		self.per_set_num = per_set_num
		self.duplicate_num = duplicate_num
		self.data_name = data_name
		self.label_name = label_name
		self.times_epoch = 0
		self.max_times_epoch = max_times_epoch
		self.hdfs_path = hdfs_path
		self.rec_name = rec_name
		self.data_type = data_type
		self.max_num_of_duplicate_set = max_num_of_duplicate_set
		self.i = 0
		self.j = 0
		self.end = 0
		#self.data = np.ones((self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]), dtype='float32')
		#self.label = np.zeros((self.batch_size, ), dtype=np.float32)

	def _shuffle(self):
		random.shuffle(self.train_list)

	@property
	def provide_data(self):
		return [(self.data_name, (self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]))]

	@property
	def provide_label(self):
		return [(self.label_name, (self.batch_size, ))]

	@property
	def batchsize(self):
		return self.batch_size

	def get_total_list(self):
		self.train = mx.io.ImageRecordIter(
			path_imgrec = self.hdfs_path + self.rec_name,
			data_shape  = self.data_shapes,
			#batch_size  = self.batch_size,
			batch_size  = 100,
			#shuffle     = 0,                 
			round_batch = 0,
			#scale       = 0.007843137,
			#mean_r      = 127.5,
			#preprocess_threads = 2,
		)
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

		sorted_label_img = sorted(label_img.items(), key = lambda x:int(x[0].split(' ')[0])) #label = sorted_label_img[i][0]   img = sorted_label_img[i][1] 
		self.sorted_label_img = np.array(sorted_label_img, dtype=object)
		print "total number is " + str(len(self.sorted_label_img))
		# import pdb; pdb.set_trace()

	def get_val_list(self):
		data_iter = mx.io.ImageRecordIter(
			path_imgrec = self.hdfs_path + self.rec_name,
			data_shape  = (3, 112, 96),
			batch_size  = 10,
			shuffle     = 0,  
			round_batch = 0,
			scale       = 0.007843137,
			mean_r      = 127.5,
			preprocess_threads = 1,
		)
		data_iter.reset()
		self.val_label_img = {} 
		while data_iter.iter_next():
			img_batch = data_iter.getdata()[0].asnumpy()
			label_batch = data_iter.getlabel().asnumpy()
			for i in range(len(label_batch)):
				label = int(label_batch[i])
				self.val_label_img.setdefault(label, [])
				self.val_label_img[label].append(img_batch[i])

	def val_next(self):
		if self.iter_next():
			val_data = []
			val_label = []
			val_num = 0
			for i in range(self.i, len(self.val_label_img)):
				img_data = np.array(self.val_label_img[i])
				
				for j in range(self.j, len(img_data)/self.per_set_num):
					# if j==(len(img_data)/self.per_set_num-1) and len(img_data)%self.per_set_num!=0:
					# 	self.j = 0
					# 	break
					# elif j==(len(img_data)/self.per_set_num-1) and len(img_data)%self.per_set_num==0:
					# 	val_data.append(img_data[j*self.per_set_num:])
					# 	label_tmp = np.ones((self.per_set_num, ))*i
					# 	val_label.append(label_tmp)
					# 	val_num += 1
					# 	self.j = 0
					# elif j==100:
					# 	self.j = 0
					# 	break
					# else:
					val_data.append(img_data[j*self.per_set_num:(j+1)*self.per_set_num])
					label_tmp = np.ones((self.per_set_num, ))*i
					val_label.append(label_tmp)
					val_num += 1
					if j==(len(img_data)/self.per_set_num-1):
						self.j = 0
					if j==30:
						self.j = 0
						break
					if val_num == self.set_num:
						self.j = j
						break
				if val_num == self.set_num:
					self.i = i
					break
			val_data = np.concatenate(val_data)
			val_label = np.concatenate(val_label)
			# import pdb; pdb.set_trace()
			
			if val_label.shape[0] != self.set_num*self.per_set_num:
				val_num = 0
				lack_num = self.set_num*self.per_set_num-val_label.shape[0]
				lack_data = []
				lack_label = []
				for i in range(0, len(self.val_label_img)):
					img_data = np.array(self.val_label_img[i])
					for j in range(0, len(img_data)/self.per_set_num):
						lack_data.append(img_data[j*self.per_set_num:(j+1)*self.per_set_num])
						label_tmp = np.ones((self.per_set_num, ))*i
						lack_label.append(label_tmp)
						val_num += 1
						if val_num == lack_num/self.per_set_num:
							break
					if val_num == lack_num/self.per_set_num:
						break
				# import pdb; pdb.set_trace()
				lack_data = np.concatenate(lack_data)
				lack_label = np.concatenate(lack_label)
				val_data = np.concatenate((val_data, lack_data))
				val_label = np.concatenate((val_label, lack_label))
				self.end = 1

			self.data = [mx.nd.array(val_data)]
			self.label = [mx.nd.array(val_label)]
			return DataBatch(data=self.data, label=self.label)
		else:
			raise StopIteration


	def reset(self):
		if self.data_type == 'train':
			if self.times_epoch == 0:
				self.get_total_list()
				self.get_label_idx()
			else:
				self.times_epoch = 0
		elif self.data_type == 'val':
			self.i = 0
			self.j = 0
			if self.end == 0:
				self.get_val_list()
			else:
				self.end = 0
		#tmp_list[i] tuple -> list
		'''
		self.tmp_list = []
		for i in range(len(self.sorted_label_img)):
			self.tmp_list.append([self.sorted_label_img[i][0], self.sorted_label_img[i][1]])
			#self.tmp_list[i].append(1)
		'''
	def iter_next(self):
		if self.data_type == 'train':
			# if self.get_label_idx():
			# 	return self.max_times_epoch > self.times_epoch
			# else:
			# 	return False
			return self.max_times_epoch > self.times_epoch
		elif self.data_type == 'val':
			if self.end==1:
				return False
			else:
				return True

	def next(self):
		if self.data_type == 'train':
			return self.train_next()
		elif self.data_type == 'val':
			return self.val_next()

	def get_label_idx(self):
		self.label_idx = {}
		if self.sorted_label_img.any():
			tmp = int(self.sorted_label_img[0][0].split(' ')[0])  #the first one
			self.label_idx[tmp] = 0
			for i in range(len(self.sorted_label_img)):
				cur_label = int(self.sorted_label_img[i][0].split(' ')[0])
				if tmp != cur_label:
					tmp = cur_label
					self.label_idx[cur_label] = i
			return True
		else:
			return False

	def get_train_list(self):
		# random_list = random.sample(self.label_idx.keys(), self.set_num)  #label list
		# if self.duplicate_num==0:
		# 	random_list = random_list
		# else:
		# 	duplicate_list = random.sample(random_list, self.duplicate_num)
		# 	duplicate_tmp = 0
		# 	for i in range(len(random_list)):
		# 		if random_list[i] in duplicate_list:
		# 			continue
		# 		else:
		# 			random_list[i] = duplicate_list[duplicate_tmp]
		# 			duplicate_tmp += 1
		# 			if duplicate_tmp == len(duplicate_list):
		# 				break

		random_list = []
		list_tmp = random.sample(self.label_idx.keys(), self.set_num)
		duplicate_list = set(random.sample(list_tmp, self.duplicate_num))
		for clabel in duplicate_list:
			dup_num = random.randint(2, self.max_num_of_duplicate_set)
			random_list.extend([clabel] * dup_num)
		for clabel in list_tmp:
			if len(random_list) >= self.set_num:
				random_list = random_list[0:self.set_num]
				break
			if clabel in duplicate_list:
				continue
			else:
				random_list.append(clabel)

		'''
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
		'''
		total_list = []
		# len(self.label_idx) = 53950
		for i in range(self.set_num):
			if random_list[i] == max(self.label_idx.keys()):  #the last one
				per_random_list = [random.randint(self.label_idx[random_list[i]], len(self.sorted_label_img)-1) for _ in range(self.per_set_num)]
			else:
				k = 0
				while not(random_list[i]+k+1 in self.label_idx.keys()):
					k += 1
				per_random_list = [random.randint(self.label_idx[random_list[i]], self.label_idx[random_list[i]+k+1]-1) for _ in range(self.per_set_num)] 
			total_list.extend(per_random_list)

		self.train_list = total_list
		#import pdb; pdb.set_trace()
	
	def save_img(self, score, nbatch, epoch):
		random_set_num = 10
		data = self.data[0].asnumpy()
		#import pdb; pdb.set_trace()
		plt.figure(num='train_img',figsize=(2*self.per_set_num,20))
		for i in range(random_set_num):
			for j in range(self.per_set_num):
				#import pdb; pdb.set_trace()
				plt.subplot(random_set_num, self.per_set_num, i*self.per_set_num+j+1)
				score_float = '%.4f' %score[i*self.per_set_num+j][0]
				plt.title(score_float, fontsize=20)
				data_mean = data[i*self.per_set_num+j]*127.5
				data_plus = data_mean
				data_plus[0,:] = data_mean[0,:]+127.5
				data_origin = data_plus.astype('uint8').transpose((1, 2, 0))
				plt.imshow(data_origin)
				plt.axis('off')
		#plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)
		plt.tight_layout()
		img_name = "epoch-"+str(epoch)+"_"+"batch-"+str(nbatch)+".png"
		plt.savefig("epoch-"+str(epoch)+"_"+"batch-"+str(nbatch)+".png")
		# os.execl("/usr/bin/env", "sh", "./test.sh", "epoch-"+str(epoch)+"_"+"batch-"+str(nbatch)+".png", "hdfs://hobot-bigdata/user/hao01.wang/")
		#print "Save Image OK"


	def train_next(self):
		self.times_epoch += 1

		if self.iter_next():
			self.get_train_list()
			data = np.ones((self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]), dtype='float32')
			label = np.zeros((self.batch_size, ), dtype=np.float32)
			#self._shuffle()
			'''
			if self.duplicate_num==0:
				for i in range(len(self.train_list)):
					prob = np.random.uniform()
					if prob <= 0.30:
						data_origin = self.sorted_label_img[self.train_list[i]][1].astype('float32').transpose((1, 2, 0))
						kernel, anchor = motion_blur(random.randint(15, 40), random.randint(20, 60))
						self.data[i] = cv2.filter2D(data_origin, -1, kernel, anchor=anchor).transpose((2, 0, 1))
					else:
						self.data[i] = self.sorted_label_img[self.train_list[i]][1].astype('float32')

			else:
				for i in range(len(self.train_list)):
					if i in self.duplicate_clean_position:
						self.data[i] = self.sorted_label_img[self.train_list[i]][1].astype('float32')
					elif i in self.duplicate_noise_position:
						prob = np.random.uniform()
						if prob <= 0.80:
							data_origin = self.sorted_label_img[self.train_list[i]][1].astype('float32').transpose((1, 2, 0))
							kernel, anchor = motion_blur(random.randint(20, 60), random.randint(20, 60))
							self.data[i] = cv2.filter2D(data_origin, -1, kernel, anchor=anchor).transpose((2, 0, 1))
						else:
							self.data[i] = self.sorted_label_img[self.train_list[i]][1].astype('float32')
					elif i in self.others_position:
						prob = np.random.uniform()
						if prob <= 0.30:
							data_origin = self.sorted_label_img[self.train_list[i]][1].astype('float32').transpose((1, 2, 0))
							kernel, anchor = motion_blur(random.randint(20, 60), random.randint(20, 60))
							self.data[i] = cv2.filter2D(data_origin, -1, kernel, anchor=anchor).transpose((2, 0, 1))
						else:
							self.data[i] = self.sorted_label_img[self.train_list[i]][1].astype('float32')

			occlusion_aug(self.batch_size, self.data_shapes, max_w=60, max_h=100, 
							min_prob=0.0, max_prob=0.3, img=self.data)
			'''
			'''
			for i in range(len(data)):
				cv2.imwrite('/data-sdc/hao01.wang/longhu/'+str(i)+' '+str(self.sorted_label_img[self.train_list[i]][0].split(' ')[0])+'.jpg', cv2.cvtColor(self.sorted_label_img[self.train_list[i]][1].astype('uint8').transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))
				#print i
			import pdb; pdb.set_trace()
			'''
			#import pdb; pdb.set_trace()
			
			for i in range(self.batch_size):
				data_aug = self.sorted_label_img[self.train_list[i]][1].astype('float32')
				data_mean = data_aug
				data_mean[0,:] = data_aug[0,:]-127.5
				data_scale = data_mean/127.5

				data[i] = data_scale
				label[i] = int(self.sorted_label_img[self.train_list[i]][0].split(' ')[0])

			# longhu_train_data = np.load('longhu_train_data.npy')
			# longhu_train_label = np.load('longhu_train_label.npy')
			# data[0:50] = longhu_train_data
			# label[0:50] = longhu_train_label
			#import pdb; pdb.set_trace()
			
			self.data = [mx.nd.array(data)]
			self.label = [mx.nd.array(label)]
			#import pdb; pdb.set_trace()
			#self.train_list = list(set(self.train_list))

			'''
			#delete samples which are trained already
			del_num = 0
			for i in range(len(self.train_list)):
				self.tmp_list[self.train_list[i]][2] = 0
			for i in range(len(self.tmp_list)):
				if self.tmp_list[i-del_num][2] == 0:
					del self.tmp_list[i-del_num]
					del_num += 1
			'''
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
