import mxnet as mx
import logging
import dataiter_custom
import custom_layer
import argparse
import os
import numpy as np

import sys 
sys.path.append("../center_model/center_three_sets/")
import metriclearn
import train_info
import train_model

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
	args = parse_args()

	conv1 = mx.sym.Convolution(data=data, num_filter=64, kernel=(7,7), stride=(2,2), pad=(3,3), name='conv1_s', no_bias=False)
	pool1 = mx.sym.Pooling(data=conv1, pool_type="max", kernel=(3,3), stride=(2,2), name='pool1_s')
	conv2 = mx.sym.Convolution(data=pool1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv1_sss1', no_bias=False)
	conv3 = mx.sym.Convolution(data=conv2, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv2_s', no_bias=False)
	pool2 = mx.sym.Pooling(data=conv3, pool_type="avg", kernel=(7,7), stride=(7,7), name='pool_s') #(batch_size, 64, 3, 3)
	fc1 = mx.sym.FullyConnected(data=pool2, num_hidden=3, name='fc1_s') #(batch_size, 3)
	act1 = mx.sym.Activation(data=fc1, name='relu1', act_type='relu')
	fc2 = mx.sym.FullyConnected(data=act1, num_hidden=1, name='fc2_s')
	score = mx.sym.Activation(data=fc2, act_type='sigmoid', name='score')
	score_l1 = mx.sym.Custom(data=score, set_num=args.set_num, per_set_num=args.per_set_num, ctx=ctx[0], 
				op_type='set_l1_norm', name='score_l1')

	return score_l1

def parse_args():
	parser = argparse.ArgumentParser(description='Set-to-Set Network')
	# training
	parser.add_argument('--set_num', help='input set number', default=30, type=int)
	parser.add_argument('--per_set_num', help='the number for per set', default=5, type=int)
	parser.add_argument('--duplicate_num', help='the number of duplicate set for a batch', default=5, type=int)
	parser.add_argument('--gpus', help='GPU device to train with', default='1', type=str)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
	args = parse_args()

	data_shapes = (3, 112, 96)
	fileiter = dataiter_custom.FileIter(data_shapes=data_shapes, set_num=args.set_num, 
					per_set_num=args.per_set_num, duplicate_num=args.duplicate_num)
	fileiter.reset()

	ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

	label = mx.sym.var("label")
	arg_params, aux_params, all_layers = load_model()
	data = all_layers['data']
	feature_output = all_layers['fc5_xaiver_256_output']
	score = tiny_network(data)
	feature_set = mx.sym.Custom(score=score, feature=feature_output, data=data, label=label, 
				set_num=args.set_num, per_set_num=args.per_set_num, ctx=ctx[0],
				op_type='batch_pool')
	metric_cost = train_info.get_cost(5*16)

	score_model = train_info.get_model(devs=ctx, network=feature_set, 
					cost=metric_cost, num_epoch=3, lr=0.0002, wd=0.00001,
					arg_params=arg_params, aux_params=aux_params)

	score_model.fit(X=fileiter,
			#eval_data=val,
			train_metric=[mx.metric.CustomMetric(feval=lambda label, pred: np.sum(pred) / pred.shape[0])],
			eval_metric=[train_model.NdcgEval(10), train_model.LFWVeri(0.0, 30.0, 300)],
			# eval_metric        = [LFWVeri(0.0, 5.0, 1000)],
			fixed_param_names=all_layers.list_arguments()[1:],
			#kvstore=kv,
			batch_end_callback=mx.callback.Speedometer(args.per_set_num*args.set_num, 1),
			#epoch_end_callback=checkpoint,
			eval_first=False,
			#l2_reg=args.l2_reg,
			#svd_epoches_period=args.do_svd_period,
			#svd_param_name=last_fc_name
			)
