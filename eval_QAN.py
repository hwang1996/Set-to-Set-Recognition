import custom_layer 
import numpy as np
import mxnet as mx
import sys 
sys.path.append("../center_model/center_three_sets/")
import metriclearn

json_file = 'model-symbol.json'
model_file = 'model_adam_no_noise_l2_0.008-0037.params'
ctx = mx.gpu(1)

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

score = sym.get_internals()['score_output']
mod_score = mx.mod.Module(symbol=score, label_names=None, context=mx.gpu(2))
mod_score.bind(for_training=False, data_shapes=[('data', (10,3,112,96))])
mod_score.set_params(arg_params, aux_params)

feature = sym.get_internals()['fc5_xaiver_256_output']
mod_fea = mx.mod.Module(symbol=feature, label_names=None, context=mx.gpu(2))
mod_fea.bind(for_training=False, data_shapes=[('data', (10,3,112,96))])
mod_fea.set_params(arg_params, aux_params)

data_iter = mx.io.ImageRecordIter(
			path_imgrec = 'hangzhougongan_test_noshuffle.rec',
			data_shape  = (3, 112, 96),
			batch_size  = 10,
			shuffle     = 0,  
			round_batch = 0,
			scale       = 0.007843137,
			mean_r      = 127.5,
			preprocess_threads = 1,
		)
data_iter.reset()

label_fea = {}
label_score = {}

while data_iter.iter_next():
	img_batch = data_iter.getdata()
	batch = mx.io.DataBatch(img_batch, [])
	label_batch = data_iter.getlabel().asnumpy()
	mod_score.forward(batch)
	mod_fea.forward(batch)
	score_output = mod_score.get_outputs()[0].asnumpy()
	feature_output = mod_fea.get_outputs()[0].asnumpy()
	for i in range(len(label_batch)):
		label = int(label_batch[i])
		label_fea.setdefault(label, [])
		label_score.setdefault(label, [])
		label_fea[label].append(feature_output[i])
		label_score[label].append(score_output[i])

label_eval = []
set_fea_eval = []
#QAN
for i in range(len(label_score)):
	single_score = np.array(label_score[i])
	single_fea = np.array(label_fea[i])
	if len(label_score[i])==1 or len(label_score[i])==2 or \
	   len(label_score[i])==3 or len(label_score[i])==4:
		set_fea_tmp = np.sum(single_fea*single_score, axis=0)
		score_sum = np.sum(single_score)
		set_fea = set_fea_tmp/score_sum
		label_eval.append(i)
		set_fea_eval.append(set_fea)
	else:
		set_fea_tmp = np.sum(single_fea[0:3]*single_score[0:3], axis=0)
		score_sum = np.sum(single_score[0:3])
		set_fea_1 = set_fea_tmp/score_sum
		label_eval.append(i)
		set_fea_eval.append(set_fea_1)

		set_fea_tmp = np.sum(single_fea[3:]*single_score[3:], axis=0)
		score_sum = np.sum(single_score[3:])
		set_fea_2 = set_fea_tmp/score_sum
		label_eval.append(i)
		set_fea_eval.append(set_fea_2)


#AvePool
# for i in range(len(label_score)):
# 	single_score = np.ones((len(label_score[i]), 1))
# 	single_fea = np.array(label_fea[i])
# 	if len(label_score[i])==1 or len(label_score[i])==2 or \
#  	   len(label_score[i])==3 or len(label_score[i])==4:
# 		set_fea_tmp = np.sum(single_fea*single_score, axis=0)
# 		score_sum = np.sum(single_score)
# 		set_fea = set_fea_tmp/score_sum
# 		label_eval.append(i)
# 		set_fea_eval.append(set_fea)
# 	else:
# 		set_fea_tmp = np.sum(single_fea[0:3]*single_score[0:3], axis=0)
# 		score_sum = np.sum(single_score[0:3])
# 		set_fea_1 = set_fea_tmp/score_sum
# 		label_eval.append(i)
# 		set_fea_eval.append(set_fea_1)

# 		set_fea_tmp = np.sum(single_fea[3:]*single_score[3:], axis=0)
# 		score_sum = np.sum(single_score[3:])
# 		set_fea_2 = set_fea_tmp/score_sum
# 		label_eval.append(i)
# 		set_fea_eval.append(set_fea_2)
	
label_eval = np.array(label_eval)
set_fea_eval = np.array(set_fea_eval)
metriclearn.metric.findMetricThreshold(set_fea_eval, label_eval, set_fea_eval, label_eval)

