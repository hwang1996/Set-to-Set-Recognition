import mxnet as mx

class batch_pool(mx.operator.CustomOp):
	def __init__(self, set_num, per_set_num, ctx):
		super(batch_pool, self).__init__()
		self.set_num = int(set_num)
		self.per_set_num = int(per_set_num)
		self.ctx = ctx

	def forward(self, is_train, req, in_data, out_data, aux):
		score = in_data[0]  #batch_size x 1 <NDArray 150x1 @gpu(0)>
		feature = in_data[1]  #batch_size x 256
		data = in_data[2]
		#mx.nd.save('my_img', [data])
		label = in_data[3]

		set_fea_tmp = mx.nd.broadcast_mul(score, feature)  #batch_size x 256
		y = mx.nd.split(set_fea_tmp, axis=0, num_outputs=self.set_num) 
		set_fea = mx.nd.zeros((self.set_num, 256), self.ctx)
		for i in range(len(y)):
			set_fea[i] = mx.nd.sum(y[i], axis=0)

		set_fea_re = mx.nd.zeros((self.per_set_num*self.set_num, 256), self.ctx)
		set_fea_re = mx.nd.repeat(set_fea, repeats=self.per_set_num, axis=0)
		
		self.feature = feature
		self.set_fea = set_fea

		self.assign(out_data[0], req[0], set_fea_re)


	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		grad = mx.nd.zeros((self.per_set_num*self.set_num, 256), self.ctx)
		for i in range(self.per_set_num*self.set_num):
			grad[i] = (self.feature[i]-self.set_fea[int(i/self.per_set_num)])*out_grad[0][int(i/self.per_set_num)]
		grad = mx.nd.sum(grad, axis=1).reshape((self.per_set_num*self.set_num,1))

		self.assign(in_grad[0], req[0], grad)
		self.assign(in_grad[1], req[1], 0)
		self.assign(in_grad[2], req[2], 0)
		self.assign(in_grad[3], req[3], 0)
		
@mx.operator.register("batch_pool")
class BatchPoolProp(mx.operator.CustomOpProp):
	def __init__(self, set_num, per_set_num, ctx):
		super(BatchPoolProp, self).__init__(need_top_grad=True)
		self.set_num = set_num
		self.per_set_num = per_set_num
		self.ctx = ctx

	def list_arguments(self):
		return ['score', 'feature', 'data', 'label']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		score_shape = in_shape[0]
		feature_shape = in_shape[1]
		data_shape = in_shape[2]
		label_shape = in_shape[3]
		output_shape = in_shape[1]
		return [score_shape, feature_shape, data_shape, label_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return batch_pool(self.set_num, self.per_set_num, self.ctx)

class set_l1_norm(mx.operator.CustomOp):
	def __init__(self, set_num, per_set_num, ctx):
		super(set_l1_norm, self).__init__()
		self.set_num = int(set_num)
		self.per_set_num = int(per_set_num)
		self.ctx = ctx

	def forward(self, is_train, req, in_data, out_data, aux):
		self.x = in_data[0]
		x_abs = mx.nd.abs(self.x)
		
		self.score_set = mx.nd.zeros((self.set_num, 1), self.ctx)
		for i in range(self.set_num):
			for j in range(self.per_set_num):
				self.score_set[i] = self.score_set[i]+x_abs[i*self.per_set_num+j]
		
		score_l1 = mx.nd.zeros((self.per_set_num*self.set_num, 1), self.ctx)
		for i in range(self.per_set_num*self.set_num):
			score_l1[i] = x_abs[i]/self.score_set[int(i/self.per_set_num)]
		self.assign(out_data[0], req[0], score_l1)

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		grad = mx.nd.zeros((self.per_set_num*self.set_num, 1), self.ctx)

		for i in range(self.per_set_num*self.set_num):
			grad[i] = (self.score_set[int(i/self.per_set_num)]-mx.nd.abs(self.x)[i])/mx.nd.square(self.score_set[int(i/self.per_set_num)])

		self.assign(in_grad[0], req[0], grad*out_grad[0])

@mx.operator.register("set_l1_norm")
class SetL1NormProp(mx.operator.CustomOpProp):
	def __init__(self, set_num, per_set_num, ctx):
		super(SetL1NormProp, self).__init__(need_top_grad=True)
		self.set_num = int(set_num)
		self.per_set_num = int(per_set_num)
		self.ctx = ctx

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		output_shape = in_shape[0]
		return [data_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return set_l1_norm(self.set_num, self.per_set_num, self.ctx)
