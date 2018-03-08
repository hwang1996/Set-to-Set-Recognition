import mxnet as mx
import numpy as np
import random
import matplotlib.pyplot as plt

import os
from mxnet.io import DataIter, DataBatch


class FileIter(DataIter):
	def __init__(self, data_shapes, set_num, per_set_num, duplicate_num, data_name="data", label_name="label"):
		#duplicate_num <= set_num/2
		super(FileIter, self).__init__()
		self.batch_size = set_num*per_set_num
		self.data_shapes = data_shapes
		self.set_num = set_num
		self.per_set_num = per_set_num
		self.duplicate_num = duplicate_num
		self.data_name = data_name
		self.label_name = label_name
		self.data = mx.nd.zeros((self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]))
		self.label = mx.nd.zeros((self.batch_size, ))

		self.train = mx.io.ImageRecordIter(
			path_imgrec = '../hangzhougongan_train_noshuffle.rec',
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

		self.sorted_label_img = sorted(label_img.items(), key = lambda x:int(x[0].split(' ')[0])) #label = sorted_label_img[i][0]   img = sorted_label_img[i][1] 

	def reset(self):
		self.get_total_list()
		#tmp_list[i] tuple -> list
		self.tmp_list = []
		for i in range(len(self.sorted_label_img)):
			self.tmp_list.append([self.sorted_label_img[i][0], self.sorted_label_img[i][1]])
			self.tmp_list[i].append(1)

	def iter_next(self):
		self.get_label_idx()
		return len(self.label_idx) >= self.set_num

	def get_label_idx(self):
		self.label_idx = {}
		tmp = int(self.tmp_list[0][0].split(' ')[0])  #the first one
		self.label_idx[tmp] = 0
		for i in range(len(self.tmp_list)):
			cur_label = int(self.tmp_list[i][0].split(' ')[0])
			if tmp != cur_label:
				tmp = cur_label
				self.label_idx[cur_label] = i

	def get_train_list(self):
		random_list = random.sample(self.label_idx.keys(), self.set_num)  #label list
		duplicate_list = random.sample(random_list, self.duplicate_num)
		duplicate_tmp = 0
		for i in range(len(random_list)):
			if random_list[i] in duplicate_list:
				continue
			else:
				random_list[i] = duplicate_list[duplicate_tmp]
				duplicate_tmp += 1
				if duplicate_tmp == len(duplicate_list):
					break

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

	def next(self):
		if self.iter_next():
			self.get_train_list()
			data = np.zeros((self.batch_size, self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]), dtype='float32')
			label = np.zeros((self.batch_size, ))
			#self._shuffle()

			for i in range(len(self.train_list)):
				data_origin = self.tmp_list[self.train_list[i]][1].astype('float32')
				data_mean = data_origin
				data_mean[0,:] = data_origin[0,:]-127.5
				data_scale = data_mean/127.5
				#Gaussian Noise 
				mu, sigma = 0, 0.1
				noise = np.random.normal(mu, sigma, (self.data_shapes[0], self.data_shapes[1], self.data_shapes[2]))
				data_scale = data_scale + noise

				data[i] = data_scale
				label[i] = int(self.tmp_list[self.train_list[i]][0].split(' ')[0])

			self.data = [mx.nd.array(data)]
			self.label = [mx.nd.array(label)]

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



