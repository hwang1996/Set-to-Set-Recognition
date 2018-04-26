import cv2
import mxnet as mx
import logging
import dataiter_custom
import custom_layer
import resnet
import argparse
import os
import numpy as np

import sys 
sys.path.append("../center_model/center_three_sets/")
import metriclearn
import train_info
import train_model

def load_model():
	args = parse_args()
	model_path = args.hdfs_path

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

	conv2 = mx.sym.Convolution(data=pool1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv2_s', no_bias=False)
	act1  = mx.sym.Activation(data=conv2, name='relu1', act_type='relu')

	conv3 = mx.sym.Convolution(data=act1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv3_s', no_bias=False)
	act2  = mx.sym.Activation(data=conv3, name='relu2', act_type='relu')
	pool2 = mx.sym.Pooling(data=act2, pool_type="max", kernel=(2,2), stride=(1,1), name='pool2_s') #(batch_size, 64, 3, 3)

	conv4 = mx.sym.Convolution(data=pool2, num_filter=128, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv4_s', no_bias=False)
	act3  = mx.sym.Activation(data=conv4, name='relu3', act_type='relu')

	conv5 = mx.sym.Convolution(data=act3, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_s', no_bias=False)
	act4  = mx.sym.Activation(data=conv5, name='relu4', act_type='relu')
	pool3 = mx.sym.Pooling(data=act4, pool_type="avg", kernel=(7,7), stride=(7,7), name='pool3_s')

	fc1   = mx.sym.FullyConnected(data=pool3, num_hidden=100, name='fc1_s') #(batch_size, 3)
	act5  = mx.sym.Activation(data=fc1, name='relu5', act_type='relu')
	fc2   = mx.sym.FullyConnected(data=act5, num_hidden=1, name='fc2_s')
	score = mx.sym.Activation(data=fc2, act_type='sigmoid', name='score')

	#score_l1 = mx.sym.Custom(data=score, set_num=args.set_num, per_set_num=args.per_set_num, ctx=ctx[0], 
							#op_type='set_l1_norm', name='score_l1')

	return score

def parse_args():
	parser = argparse.ArgumentParser(description='Set-to-Set Network')
	# training
	parser.add_argument('--set_num', help='input set number', default=400, type=int)
	parser.add_argument('--per_set_num', help='the number for per set', default=3, type=int)
	parser.add_argument('--duplicate_num', help='the number of duplicate set for a batch', default=100, type=int)
	parser.add_argument('--max_num_of_duplicate_set', default=5, type=int)
	parser.add_argument('--gpus', help='GPU device to train with', default='0,1,2,3', type=str)
	parser.add_argument('--model_prefix', help='model prefix to save model', default='model_0000', type=str)
	parser.add_argument('--hdfs_path', help='path in hdfs', default='hdfs://hobot-bigdata/user/hao01.wang/', type=str)
	parser.add_argument('--train_rec', help='rec for train', default='LongHU_Track.rec', type=str)
	parser.add_argument('--val_rec', help='rec for val', default='aligned_images_DB.rec', type=str)
	parser.add_argument('--max_times_epoch', help='times of training in each epoch', default=100, type=int)
	parser.add_argument('--l2_param', help='parameter for L2 penalty', default=0.0000, type=float)
	parser.add_argument('--top_n', help='top n images in a batch -> 0.9', default=50, type=int)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	
	logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
	args = parse_args()
	ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

	if isinstance(ctx, list):
		assert args.set_num%len(ctx) == 0, 'set_num error!'
	elif isinstance(ctx, str):
		assert args.set_num%len(ctx.strip(',').split(',')) == 0, 'set_num error!'

	data_shapes = (3, 112, 96)
	fileiter_train = dataiter_custom.FileIter(
			data_shapes              = data_shapes, 
			set_num                  = args.set_num,
			per_set_num              = args.per_set_num, 
			duplicate_num            = args.duplicate_num,
			max_num_of_duplicate_set = args.max_num_of_duplicate_set,
			hdfs_path                = args.hdfs_path,
			rec_name                 = args.train_rec,
			max_times_epoch          = args.max_times_epoch,
			data_type                = 'train'
		)
	fileiter_val = dataiter_custom.FileIter(
			data_shapes              = data_shapes, 
			set_num                  = args.set_num,
			per_set_num              = args.per_set_num, 
			duplicate_num            = args.duplicate_num,
			max_num_of_duplicate_set = args.max_num_of_duplicate_set,
			hdfs_path                = args.hdfs_path,
			rec_name                 = args.val_rec,
			max_times_epoch          = args.max_times_epoch,
			data_type                = 'val'
		)

	label = mx.sym.var("label")
	arg_params, aux_params, all_layers = load_model()
	data = all_layers['data']
	feature_output = all_layers['fc5_xaiver_256_output']
	# score = tiny_network(data)
	score = resnet.get_symbol(data=data, num_classes=1, num_layers=18, image_shape=data_shapes)
	feature_set = mx.sym.Custom(score=score, feature=feature_output, data=data, label=label, 
								set_num=args.set_num, per_set_num=args.per_set_num, ctx=ctx,
								op_type='batch_pool')
	metric_cost = train_info.get_cost(5*16)

	l2_param = float(args.l2_param)
	l2_penalty_total = mx.sym.Custom(score=score, set_num=args.set_num, per_set_num=args.per_set_num,
								l2_param=l2_param, ctx=ctx, top_n=args.top_n, op_type='l2_penalty_total')

	score_model = train_info.get_model(devs=ctx, network=mx.sym.Group([feature_set, l2_penalty_total, mx.sym.BlockGrad(score)]), 
										cost=metric_cost, num_epoch=100, optimizer='adam', lr=0.0002, wd=0.00001,
										arg_params=arg_params, aux_params=aux_params)

	checkpoint = None if args.model_prefix is None else mx.callback.do_checkpoint(args.hdfs_path+args.model_prefix)

	score_model.fit(X=fileiter_train,
					eval_data=fileiter_val,
					train_metric=[mx.metric.CustomMetric(feval=lambda label, pred: np.sum(pred) / pred.shape[0])],
					#eval_metric=[train_model.NdcgEval(10), train_model.LFWVeri(0.0, 30.0, 300)],
					eval_metric=[train_model.NdcgEval(args.per_set_num)],
					fixed_param_names=all_layers.list_arguments()[1:],
					#kvstore=kv,
					batch_end_callback=mx.callback.Speedometer(args.per_set_num*args.set_num, 1),
					epoch_end_callback=checkpoint,
					eval_first=True,
					#l2_reg=args.l2_reg,
					#svd_epoches_period=args.do_svd_period,
					#svd_param_name=last_fc_name
					)
