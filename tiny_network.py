import mxnet as mx
import logging
import dataiter_custom
import os
import numpy as np

import sys 
sys.path.append("../center_model/center_three_sets/")
import metriclearn
import train_info
import train_model


class batch_pool(mx.operator.CustomOp):
	def forward(self, is_train, req, in_data, out_data, aux):
		score = in_data[0]  #batch_size x 1 <NDArray 150x1 @gpu(0)>
		feature = in_data[1]  #batch_size x 256
		data = in_data[2]
		#mx.nd.save('my_img', [data])
		label = in_data[3]

		set_fea_tmp = mx.nd.broadcast_mul(score, feature)  #batch_size x 256
		y = mx.nd.split(set_fea_tmp, axis=0, num_outputs=30) 
		set_fea = mx.nd.zeros((30, 256), mx.gpu(1))
		for i in range(len(y)):
			set_fea[i] = mx.nd.sum(y[i], axis=0)

		score_t = mx.nd.reshape(score, shape=(5, 30)) 
		score_total = mx.nd.sum(score_t, axis=0).reshape((30,1))  
		set_fea = mx.nd.broadcast_div(set_fea, score_total)
		set_fea_re = mx.nd.zeros((150, 256), mx.gpu(1))
		set_fea_re = mx.nd.repeat(set_fea, repeats=5, axis=0)
		
		self.feature = feature
		self.score_total = score_total
		self.assign(out_data[0], req[0], set_fea_re)
		#import pdb; pdb.set_trace()

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		fea = mx.nd.sum(self.feature, axis=1)
		grad = mx.nd.zeros((150, 1), mx.gpu(1))
		for i in range(150):
			grad[i] = fea[i]/self.score_total[int(i/5)]
		self.assign(in_grad[0], req[0], (mx.nd.sum(out_grad[0], axis=1)*grad).reshape((150,1)))
		self.assign(in_grad[1], req[1], 0)
		self.assign(in_grad[2], req[2], 0)
		self.assign(in_grad[3], req[3], 0)


@mx.operator.register("batch_pool")
class BatchPoolProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(BatchPoolProp, self).__init__(need_top_grad=True)

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
		return batch_pool()

class set_l1_norm(mx.operator.CustomOp):
	def forward(self, is_train, req, in_data, out_data, aux):
		self.x = in_data[0]
		x_abs = mx.nd.abs(self.x)
		self.score_set = mx.nd.sum(mx.nd.reshape(x_abs, shape=(5, 30)), axis=0)
		score_l1 = mx.nd.zeros((150, 1))
		for i in range(150):
			score_l1[i] = x_abs[i]/self.score_set[int(i/5)]
		self.assign(out_data[0], req[0], score_l1)

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		grad = mx.nd.zeros((150, 1), mx.gpu(1))

		for i in range(150):
			grad[i] = (self.x/mx.nd.abs(self.x))[i]/self.score_set[int(i/5)]

		self.assign(in_grad[0], req[0], grad*out_grad[0])

@mx.operator.register("set_l1_norm")
class SetL1NormProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(SetL1NormProp, self).__init__(need_top_grad=True)

	def list_arguments(self):
		return ['data']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		output_shape = in_shape[0]
		return [data_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return set_l1_norm()


def load_model():
	model_path = '../'

	json_file = model_path + 'models-7-symbol.json'
	model_file = model_path + 'models-7-0107.params'

	sym = mx.sym.load(json_file)
	save_dict = mx.nd.load(model_file)

	arg_params = {}
	aux_params = {}
	loaded_params = []
	for k, v in save_dict.items():
		tp, name = k.split(':', 1)
		loaded_params.append(name)
		if tp == 'arg':
			arg_params[name] = v
		if tp == 'aux':
			aux_params[name] = v

	all_layers = sym.get_internals()

	return arg_params, aux_params, all_layers

def tiny_network(data):

	conv1 = mx.sym.Convolution(data=data, num_filter=64, kernel=(7,7), stride=(2,2), pad=(3,3), name='conv1_s', no_bias=False)
	pool1 = mx.sym.Pooling(data=conv1, pool_type="max", kernel=(3,3), stride=(2,2), name='pool1_s')
	conv2 = mx.sym.Convolution(data=pool1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv1_sss1', no_bias=False)
	conv3 = mx.sym.Convolution(data=conv2, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv2_s', no_bias=False)
	pool2 = mx.sym.Pooling(data=conv3, pool_type="avg", kernel=(7,7), stride=(7,7), name='pool_s') #(batch_size, 64, 3, 3)
	fc1 = mx.sym.FullyConnected(data=pool2, num_hidden=3, name='fc1_s') #(batch_size, 3)
	act1 = mx.sym.Activation(data=fc1, name='relu1', act_type='relu')
	fc2 = mx.sym.FullyConnected(data=act1, num_hidden=1, name='fc2_s')
	score = mx.sym.Activation(data=fc2, act_type='sigmoid')
	score_l1 = mx.sym.Custom(data=score, op_type='set_l1_norm')

	return score_l1

if __name__ == '__main__':
	logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
	
	data_shapes = (3, 112, 96)
	fileiter = dataiter_custom.FileIter(data_shapes=data_shapes, set_num=30, per_set_num=5, duplicate_num=5)
	fileiter.reset()

	label = mx.sym.var("label")
	arg_params, aux_params, all_layers = load_model()
	data = all_layers['data']
	feature_output = all_layers['fc5_xaiver_256_output']
	score = tiny_network(data)
	feature_set = mx.sym.Custom(score=score, feature=feature_output, data=data, label=label, op_type='batch_pool')
	metric_cost = train_info.get_cost(5*16)

	score_model = train_info.get_model(devs=mx.gpu(1), network=feature_set, 
					cost=metric_cost, num_epoch=10, lr=0.0002, wd=0.00001,
					arg_params=arg_params, aux_params=aux_params)

	score_model.fit(X=fileiter,
			#eval_data=val,
			train_metric=[mx.metric.CustomMetric(feval=lambda label, pred: np.sum(pred) / pred.shape[0])],
			eval_metric=[train_model.NdcgEval(10), train_model.LFWVeri(0.0, 30.0, 300)],
			# eval_metric        = [LFWVeri(0.0, 5.0, 1000)],
			fixed_param_names=all_layers.list_arguments()[1:],
			#kvstore=kv,
			batch_end_callback=mx.callback.Speedometer(150, 1),
			#epoch_end_callback=checkpoint,
			eval_first=False,
			#l2_reg=args.l2_reg,
			#svd_epoches_period=args.do_svd_period,
			#svd_param_name=last_fc_name
			)
