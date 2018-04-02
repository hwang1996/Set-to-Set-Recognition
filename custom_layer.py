import mxnet as mx
import numpy as np

class batch_pool(mx.operator.CustomOp):
	def __init__(self, set_num, per_set_num, ctx):
		super(batch_pool, self).__init__()
		if isinstance(ctx, list):
			self.set_num = int(set_num) / len(ctx)
		elif isinstance(ctx, str):
			self.set_num = int(set_num) / len(ctx.strip(',').split(','))
		self.per_set_num = int(per_set_num)
		self.ctx = ctx				

	def forward(self, is_train, req, in_data, out_data, aux):
		score = in_data[0]  #batch_size x 1 <NDArray 150x1 @gpu(0)>
		feature = in_data[1]  #batch_size x 256
		data = in_data[2]
		#mx.nd.save('my_img', [data])
		label = in_data[3]
		
		self.score_sum = mx.nd.zeros((self.set_num, 1), ctx=score.context)
		for i in range(self.set_num):
			for j in range(self.per_set_num):
				self.score_sum[i] = self.score_sum[i]+score[i*self.per_set_num+j]

		set_fea_tmp = mx.nd.broadcast_mul(score, feature)  #batch_size x 256
		#print set_fea_tmp.shape
		y = mx.nd.split(set_fea_tmp, axis=0, num_outputs=self.set_num) 
		set_fea = mx.nd.zeros((self.set_num, 256), ctx=score.context)
		for i in range(len(y)):
			set_fea[i] = mx.nd.sum(y[i], axis=0)/self.score_sum[i]

		set_fea_re = mx.nd.zeros((self.per_set_num*self.set_num, 256), ctx=score.context)
		set_fea_re = mx.nd.repeat(set_fea, repeats=self.per_set_num, axis=0)
		
		self.feature = feature
		self.set_fea_re = set_fea_re
		self.score = score
		#print 'set_fea_re is '
		#print set_fea_re.asnumpy()
		self.assign(out_data[0], req[0], set_fea_re)
		#import pdb; pdb.set_trace()

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		grad = mx.nd.zeros((self.per_set_num*self.set_num, 256), ctx=out_grad[0].context)
		
		for i in range(self.set_num*self.per_set_num):
			grad[i] = (self.feature[i]*self.score_sum[int(i/self.per_set_num)] \
				- self.set_fea_re[i]*self.score_sum[int(i/self.per_set_num)])/mx.nd.square(self.score_sum[int(i/self.per_set_num)])

		grad = grad*out_grad[0]
		grad = mx.nd.sum(grad, axis=1).reshape((self.per_set_num*self.set_num,1))

		self.assign(in_grad[0], req[0], grad)
		self.assign(in_grad[1], req[1], 0)
		self.assign(in_grad[2], req[2], 0)
		self.assign(in_grad[3], req[3], 0)
		
@mx.operator.register("batch_pool")
class BatchPoolProp(mx.operator.CustomOpProp):
	def __init__(self, set_num, per_set_num, ctx):
		super(BatchPoolProp, self).__init__(need_top_grad=True)
		self.set_num = int(set_num)
		self.per_set_num = int(per_set_num)
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



class l2_penalty(mx.operator.CustomOp):
	def __init__(self, set_num, per_set_num, l2_param, ctx):
		super(l2_penalty, self).__init__()
		if isinstance(ctx, list):
			self.set_num = int(set_num) / len(ctx)
		elif isinstance(ctx, str):
			self.set_num = int(set_num) / len(ctx.strip(',').split(','))
		self.per_set_num = int(per_set_num)
		self.l2_param = float(l2_param)

	def forward(self, is_train, req, in_data, out_data, aux):
		self.score = in_data[0]
		score_set = mx.nd.split(self.score, axis=0, num_outputs=self.set_num)
		self.max_index = []
		for i in range(self.set_num):
			np_tmp = score_set[i].asnumpy()
			self.max_index.append(np.argmax(np_tmp)+i*self.per_set_num)
		
		l2_penalty_result = mx.nd.zeros((self.set_num*self.per_set_num, 1), ctx=in_data[0].context)
		self.score_max = []
		for i in range(self.set_num):
			self.score_max.append(mx.nd.max(score_set[i]))
			l2_penalty_result[self.max_index[i]] = self.l2_param * mx.nd.square(self.score_max[i]-0.95)

		self.assign(out_data[0], req[0], l2_penalty_result)

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		grad = mx.nd.zeros((self.per_set_num*self.set_num, 1), ctx=in_data[0].context)
		for i in range(self.set_num):
			grad[self.max_index[i]] = -2 * self.l2_param * mx.nd.abs(self.score_max[i]-0.95)

		self.assign(in_grad[0], req[0], grad)

@mx.operator.register("l2_penalty")
class L2PenaltyProp(mx.operator.CustomOpProp):
	def __init__(self, set_num, per_set_num, l2_param, ctx):
		super(L2PenaltyProp, self).__init__(need_top_grad=True)
		self.set_num = int(set_num)
		self.per_set_num = int(per_set_num)
		self.l2_param = float(l2_param)
		self.ctx = ctx

	def list_arguments(self):
		return ['score']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		score_shape = in_shape[0]
		output_shape = in_shape[0]
		return [score_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return l2_penalty(self.set_num, self.per_set_num, self.l2_param, self.ctx)



class set_l1_norm(mx.operator.CustomOp):
	def __init__(self, set_num, per_set_num, ctx):
		super(set_l1_norm, self).__init__()
		self.set_num = int(set_num)
		self.per_set_num = int(per_set_num)
		self.ctx = ctx

	def forward(self, is_train, req, in_data, out_data, aux):
		self.x = in_data[0]
		x_abs = mx.nd.abs(self.x)
		
		self.score_sum = mx.nd.zeros((self.set_num, 1), self.ctx)
		for i in range(self.set_num):
			for j in range(self.per_set_num):
				self.score_sum[i] = self.score_sum[i]+x_abs[i*self.per_set_num+j]
		
		score_l1 = mx.nd.zeros((self.per_set_num*self.set_num, 1), self.ctx)
		for i in range(self.per_set_num*self.set_num):
			score_l1[i] = x_abs[i]/self.score_sum[int(i/self.per_set_num)]
		self.assign(out_data[0], req[0], score_l1)

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		grad = mx.nd.zeros((self.per_set_num*self.set_num, 1), self.ctx)

		for i in range(self.per_set_num*self.set_num):
			grad[i] = (self.score_sum[int(i/self.per_set_num)]-mx.nd.abs(self.x)[i])/mx.nd.square(self.score_sum[int(i/self.per_set_num)])

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
