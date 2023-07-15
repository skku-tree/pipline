import torch
import matplotlib.pyplot as plt
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
from collections import OrderedDict
import numpy as np
import argparse
import os
import torch.nn as nn
i=0
resnet = models.resnet50(pretrained=True)
image = []
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            #print('name=',name)
            #print('x.size()=',x.size())
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            #print('outputs.size()=',x.size())
        #print('len(outputs)',len(outputs))
        return outputs, x

class ModelOutputs():
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		#print('classfier=',output.size())
		if self.cuda:
			output = output.cpu()
			output = resnet.fc(output).cuda()
		else:
			output = resnet.fc(output)
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img
	input.requires_grad = True
	return input

def show_cam_on_image(img, mask,name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam/cam_{}.jpg".format(name), np.uint8(255 * cam))
class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		#self.model.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam
class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()
		for module in self.model.named_modules():
			module[1].register_backward_hook(self.bp_relu)

	def bp_relu(self, module, grad_in, grad_out):
		if isinstance(module, nn.ReLU):
			return (torch.clamp(grad_in[0], min=0.0),)
	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)
		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		
		one_hot.backward(retain_graph=True)
		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output





 
model = models.resnet50(pretrained=True)
del model.fc
print(model)
grad_cam = GradCam(model , \
				target_layer_names = ["layer4"], use_cuda=True)
x=os.walk("/home/minseopark/바탕화면/pipline/dataset/train_set/")
for root,dirs,filename in x:
	print(filename)
for s in filename:
		image.append(cv2.imread("/home/minseopark/바탕화면/pipline/dataset/train_set/"+s,1))
		break
for img in image:
	img = np.float32(cv2.resize(img, (224, 224))) / 255
	input = preprocess_image(img)
	input.required_grad = True
	print('input.size()=',input.size())
	target_index =None

	mask = grad_cam(input, target_index)
	i=i+1 
	show_cam_on_image(img, mask,i)

	gb_model = GuidedBackpropReLUModel(model = models.resnet50(pretrained=True), use_cuda=True)
	gb = gb_model(input, index=target_index)
	if not os.path.exists('gb'):
		os.mkdir('gb')
	if not os.path.exists('camgb'):
		os.mkdir('camgb')
	utils.save_image(torch.from_numpy(gb), 'gb/gb_{}.jpg'.format(i))
	cam_mask = np.zeros(gb.shape)
	for j in range(0, gb.shape[0]):
		cam_mask[j, :, :] = mask
	cam_gb = np.multiply(cam_mask, gb)
	utils.save_image(torch.from_numpy(cam_gb), 'camgb/cam_gb_{}.jpg'.format(i))