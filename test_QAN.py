import cv2
import mxnet as mx
import custom_layer
import dataiter_custom
import numpy as np

json_file = 'model-symbol.json'
model_file = 'model-0011.params'
ctx = mx.gpu(3)

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
mod = mx.mod.Module(symbol=score, context=mx.gpu(1), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,112,96))])
mod.set_params(arg_params, aux_params)

for i in range(150):
   img = cv2.imread(str(i)+'.jpg').astype(np.float32)
   #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = cv2.resize(img, (112, 96))
   img = img.transpose((2, 1, 0))
   img[0,:] = img[0,:]-127.5
   img = img/127.5
   img = img[np.newaxis, :]
   data = mx.nd.array(img, mx.gpu(1))
   batch = mx.io.DataBatch([data], [])
   mod.forward(batch)
   print i, mod.get_outputs()[0].asnumpy()
