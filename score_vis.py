import cv2
import mxnet as mx
import custom_layer
import dataiter_custom
import numpy as np
import os

#json_file = 'model-symbol.json'
#model_file = 'model_adam_no_noise_l2_0.008-0037.params'

json_file = 'model_adam_vis_0008-symbol.json'
model_file = 'model_adam_vis_0008-0015.params'

ctx = mx.gpu(2)

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
#print sym.list_arguments()
mod = mx.mod.Module(symbol=score, label_names=None, context=ctx)
mod.bind(for_training=False, data_shapes=[('data', (1,3,112,96))])
# mod = mx.mod.Module(symbol=score, context=ctx, data_names=['data_no_align'], label_names=None)
# mod.bind(for_training=False, data_shapes=[('data_no_align', (1,3,72,72))])

mod.set_params(arg_params, aux_params)

#f = open("no_noise_test.txt", "w")

#for i in range(200):
total_num = 0
above_num = 0
for filename in os.listdir(r"../../longhu"):
# for i in range(500):
   # img = cv2.imread('../../longhu/'+str(i)+'.jpg').astype(np.float32)
   img = cv2.imread('../../longhu/'+filename).astype(np.float32)
   image = img
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   #img = cv2.resize(img, (96, 112))
   img = img.transpose((2, 0, 1))
   #print img.shape
   img[0,:] = img[0,:]-127.5
   img = img/127.5
   img = img[np.newaxis, :]
   data = mx.nd.array(img, ctx)
   batch = mx.io.DataBatch([data], [])
   mod.forward(batch)
   '''
   if mod.get_outputs()[0][0].asnumpy()[0] >= 0.9:
      above_num += 1
   total_num += 1
   '''
   cv2.imwrite('../../longhu_out/'+filename.split(' ')[0]+' '+'%.6f' %mod.get_outputs()[0][0].asnumpy()[0]+'.jpg', image)
   # cv2.imwrite('../../longhu_out/'+'%.6f' %mod.get_outputs()[0][0].asnumpy()[0]+'.jpg', image)
   #print float(mod.get_outputs()[0][0].asnumpy()[0])
   #f.write(str(i)+' '+str(mod.get_outputs()[0][0].asnumpy()[0])+'\n')
# print above_num
# print total_num
# print float(above_num)/float(total_num)
#f.close()
