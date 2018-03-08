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
		set_fea_ = mx.nd.broadcast_mul(score, feature)
		score_t = mx.nd.reshape(score, shape=(5, 30))
		score_total = mx.nd.sum(score_t, axis=0)
		set_fea = mx.nd.zeros((150, 256))
		for i in range(150):
			set_fea[i] = set_fea_[i]/score_total[int(i/5)]
		self.assign(out_data[0], req[0], set_fea)
		#import pdb; pdb.set_trace()

@mx.operator.register("batch_pool")
class BatchPoolProp(mx.operator.CustomOpProp):
	def __init__(self):
		super(BatchPoolProp, self).__init__(need_top_grad=False)

	def list_arguments(self):
		return ['score', 'feature']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shape):
		score_shape = in_shape[0]
		feature_shape = in_shape[1]
		output_shape = feature_shape
		return [score_shape, feature_shape], [output_shape], []

	def create_operator(self, ctx, shapes, dtypes):
		return batch_pool()

def load_model():
	model_path = '../'

	json_file = model_path + 'models-7-symbol.json'
	model_file = model_path + 'models-7-0107.params'
	ctx = mx.gpu(2)
	batch_size = 150
	img_shape = (3, 112, 96)

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

	fe_mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
	fe_mod.bind(for_training=False, data_shapes=[('data', (batch_size,3,112,96))])
	fe_mod.set_params(arg_params, aux_params)

	return fe_mod, all_layers

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

	return score

if __name__ == '__main__':
	logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
	
	data_shapes = (3, 112, 96)
	fileiter = dataiter_custom.FileIter(data_shapes=data_shapes, set_num=30, per_set_num=5, duplicate_num=5)
	fileiter.reset()

	label = mx.sym.var("label")
	fe_mod, all_layers = load_model()
	data = all_layers['data']
	feature_output = all_layers['fc5_xaiver_256_output']
	score = tiny_network(data)
	feature_set = mx.sym.Custom(score=score, feature=feature_output, op_type='batch_pool')
	metric_cost = train_info.get_cost(5*16)

	score_model = train_info.get_model(devs=[mx.gpu(1)], network=feature_set, 
					   cost=metric_cost, num_epoch=100, lr=0.0002, wd=0.00001)

	score_model.fit(X=fileiter,
			#eval_data=val,
			train_metric=[mx.metric.CustomMetric(feval=lambda label, pred: np.sum(pred) / pred.shape[0])],
			eval_metric=[train_model.NdcgEval(10), train_model.LFWVeri(0.0, 30.0, 300)],
			# eval_metric        = [LFWVeri(0.0, 5.0, 1000)],
			fixed_param_names=all_layers.list_arguments(),
			#kvstore=kv,
			batch_end_callback=mx.callback.Speedometer(150, 10),
			#epoch_end_callback=checkpoint,
			eval_first=False,
			#l2_reg=args.l2_reg,
			#svd_epoches_period=args.do_svd_period,
			#svd_param_name=last_fc_name
			)
