import custom_layer 
import numpy as np
import mxnet as mx
import sys 
sys.path.append("../center_model/center_three_sets/")
import metriclearn

json_file = 'model_resnet_set_2800_0000-symbol.json'
model_file = 'model_resnet_set_2800_0000-0021.params'
ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
# ctx = mx.gpu(0)

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
mod_score = mx.mod.Module(symbol=score, label_names=None, context=ctx)
mod_score.bind(for_training=False, data_shapes=[('data', (10,3,112,96))])
mod_score.set_params(arg_params, aux_params)

feature = sym.get_internals()['fc5_xaiver_256_output']
mod_fea = mx.mod.Module(symbol=feature, label_names=None, context=ctx)
mod_fea.bind(for_training=False, data_shapes=[('data', (10,3,112,96))])
mod_fea.set_params(arg_params, aux_params)

data_iter = mx.io.ImageRecordIter(
			# path_imgrec = 'hdfs://hobot-bigdata/user/xin.wang/data/face/face_affine/affine_similarity/h_112_w_96/val/hobot_238.rec',
			path_imgrec = 'hdfs://hobot-bigdata/user/xin.wang/data/face/face_affine/affine_similarity/h_112_w_96/val/hangzhougongan_test_20171123.rec',
			# path_imgrec = 'hdfs://hobot-bigdata/user/hao01.wang/aligned_images_DB.rec',
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
feature_test = []
label_test = []

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
		# label_score[label].append(score_output[i])
		label_score[label].append(np.ones(1)*0.5)

		# feature_test.append(feature_output[i])
		# label_test.append(np.array(label))
		
		# try:
		# 	label_score[label][0]
		# except IndexError:
		# 	label_score[label].append(np.zeros(1))
		# else:
		# 	if label_score[label][len(label_score[label])-1]==0:
		# 		label_score[label].append(np.ones(1))
		# 	else:
		# 		label_score[label].append(np.zeros(1))

label_eval = []
set_fea_eval = []
end = 0
per_set_num = 3

#custom
set_num = 2400

out_loop = 0
in_loop = 0
val_num = 0
loop_time = 0
while end==0:
	val_data = []
	val_label = []
	val_num = 0
	for i in range(out_loop, len(label_score)):
		single_score = np.array(label_score[i])
		single_fea = np.array(label_fea[i])
		for j in range(in_loop, len(single_score)/per_set_num):
			set_fea_tmp = np.sum(single_fea[j*per_set_num:(j+1)*per_set_num]*single_score[j*per_set_num:(j+1)*per_set_num], axis=0)
			score_sum = np.sum(single_score[j*per_set_num:(j+1)*per_set_num])

			set_fea = set_fea_tmp/score_sum
			val_label.append(i)
			val_data.append(set_fea)
			val_num += 1
			if j==(len(single_score)/per_set_num-1):
				in_loop = 0
			if j==30:
				in_loop = 0
				break
			if val_num == set_num:
				in_loop = j
				break
		if val_num == set_num:
			out_loop = i
			break

	# import pdb; pdb.set_trace()
	val_label = np.array(val_label)
	val_data = np.array(val_data)
	if loop_time == 0:
		label_eval = val_label
		set_fea_eval = val_data
	else:
		label_eval = np.concatenate((label_eval, val_label))
		set_fea_eval = np.concatenate((set_fea_eval, val_data))

	if val_label.shape[0] != set_num:
		val_num = 0
		lack_num = set_num-val_label.shape[0]
		lack_data = []
		lack_label = []
		# import pdb; pdb.set_trace()
		for i in range(0, len(label_score)):
			single_score = np.array(label_score[i])
			single_fea = np.array(label_fea[i])
			for j in range(in_loop, len(single_score)/per_set_num):
				set_fea_tmp = np.sum(single_fea[j*per_set_num:(j+1)*per_set_num]*single_score[j*per_set_num:(j+1)*per_set_num], axis=0)
				score_sum = np.sum(single_score[j*per_set_num:(j+1)*per_set_num])

				set_fea = set_fea_tmp/score_sum
				lack_label.append(i)
				lack_data.append(set_fea)
				val_num += 1
				if val_num == lack_num:
					break
			if val_num == lack_num:
				break
		lack_data = np.array(lack_data)
		lack_label = np.array(lack_label)
		label_eval = np.concatenate((label_eval, lack_label))
		set_fea_eval = np.concatenate((set_fea_eval, lack_data))
		end = 1
	loop_time += 1

print loop_time
print label_eval.shape
print set_fea_eval.shape
# import pdb; pdb.set_trace()
metriclearn.metric.findMetricThreshold(set_fea_eval, label_eval, set_fea_eval, label_eval)
# print np.array(feature_test).shape
# metriclearn.metric.findMetricThreshold(np.array(feature_test), np.array(label_test), np.array(feature_test), np.array(label_test))
# metriclearn.metric.CalAccuracyTopN_MPI(np.array(feature_test), np.array(label_test), np.array(feature_test), np.array(label_test), topN_l=[1, 5, 10, 20], leave_one_out=True)


'''  no loop
import custom_layer
import numpy as np
import mxnet as mx
import sys
sys.path.append("../center_model/center_three_sets/")
import metriclearn

json_file = 'model_resnet_set_2800_0000-symbol.json'
model_file = 'model_resnet_set_2800_0000-0021.params'
ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
# ctx = mx.gpu(0)

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
mod_score = mx.mod.Module(symbol=score, label_names=None, context=ctx)
mod_score.bind(for_training=False, data_shapes=[('data', (10,3,112,96))])
mod_score.set_params(arg_params, aux_params)

feature = sym.get_internals()['fc5_xaiver_256_output']
mod_fea = mx.mod.Module(symbol=feature, label_names=None, context=ctx)
mod_fea.bind(for_training=False, data_shapes=[('data', (10,3,112,96))])
mod_fea.set_params(arg_params, aux_params)

data_iter = mx.io.ImageRecordIter(
                        # path_imgrec = 'hdfs://hobot-bigdata/user/xin.wang/data/face/face_affine/affine_similarity/h_112_w_96/val/hobot_238.rec',
                        path_imgrec = 'hdfs://hobot-bigdata/user/xin.wang/data/face/face_affine/affine_similarity/h_112_w_96/val/hangzhougongan_test_20171123.rec',
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
feature_test = []
label_test = []

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
                # label_score[label].append(np.ones(1)*0.5)

                feature_test.append(feature_output[i])
                label_test.append(np.array(label))

                # try:
                #       label_score[label][0]
                # except IndexError:
                #       label_score[label].append(np.zeros(1))
                # else:
                #       if label_score[label][len(label_score[label])-1]==0:
                #               label_score[label].append(np.ones(1))
                #       else:
                #               label_score[label].append(np.zeros(1))

label_eval = []
set_fea_eval = []
per_set_num = 3
# import pdb; pdb.set_trace()

for i in range(len(label_score)):
        single_score = np.array(label_score[i])
        single_fea = np.array(label_fea[i])
        for j in range(len(single_score)/per_set_num):
                # import pdb; pdb.set_trace()
                # if j==(len(single_score)/per_set_num-1) and len(single_score)%per_set_num!=0:
                #       num += 1
                #       break
                # elif j==(len(single_score)/per_set_num-1) and len(single_score)%per_set_num==0:
                #       import pdb; pdb.set_trace()
                #       set_fea_tmp = np.sum(single_fea[j*per_set_num:]*single_score[j*per_set_num:], axis=0)
                #       score_sum = np.sum(single_score[j*per_set_num:])
                #       # import pdb; pdb.set_trace()
                #       set_fea = set_fea_tmp/score_sum
                #       label_eval.append(i)
                #       set_fea_eval.append(set_fea)
                # elif j==10:
                #       break
                # # else:
                #       import pdb; pdb.set_trace()
                set_fea_tmp = np.sum(single_fea[j*per_set_num:(j+1)*per_set_num]*single_score[j*per_set_num:(j+1)*per_set_num], axis=0)
                score_sum = np.sum(single_score[j*per_set_num:(j+1)*per_set_num])

                set_fea = set_fea_tmp/score_sum
                label_eval.append(i)
                set_fea_eval.append(set_fea)

label_eval = np.array(label_eval)
set_fea_eval = np.array(set_fea_eval)

print label_eval.shape
print set_fea_eval.shape
# import pdb; pdb.set_trace()
metriclearn.metric.findMetricThreshold(set_fea_eval, label_eval, set_fea_eval, label_eval)
'''
