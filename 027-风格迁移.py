"""

@file   : 027-风格迁移.py

@author : xiaolu

@time   : 2019-06-17

"""
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave



# 定义目标图片和风格图片
target_image_path = 'sjtu.jpg'
style_reference_image_path = 'candy.jpg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width*img_height/height)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # print(img)
    # print(img.shape)   # (1, 400, 533, 3)
    img = vgg19.preprocess_input(img)     # 去中心化
    # print(img.shape)   # (1, 400, 533, 3)
    # print(img)
    return img


def deprocess_image(x):
    # 这些操作相当于vgg19.preprocess_input()的逆操作
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]   # BGR->RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 定义输入
target_image = K.constant(preprocess_image(target_image_path))    # 原始图
style_reference_image = K.constant(preprocess_image(style_reference_image_path))    # 风格图
combination_image = K.placeholder((1, img_height, img_width, 3))   # 生成的图
input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)   # 形成的输入张量


# 加载VGG19模型
model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
print("Model loaded.")


# 定义内容损失
def content_loss(base, combination):
    # 就是相当于标准差
    return K.sum(K.square(combination - base))


# 定义风格损失
def gram_matrix(x):
    # 1. gram矩阵的计算
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    # 2.计算风格损失
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S-C)) / (4. * (channels ** 2) * (size ** 2))


# 总变差损失
def total_variation_loss(x):
    a = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1, :])
    b = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, :img_height-1, 1:, :])
    return K.sum(K.pow(a+b, 1.25))


# 总损失
output_dict = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = 'block5_conv2'  # 这一层计算内容损失

style_layers = ['block5_conv2',
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']   # 这些层计算风格损失

# 定义三种损失的权重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025


loss = K.variable(0.)
# 计算内容损失
layer_features = output_dict[content_layer]
target_image_features = layer_features[0, :, :, :]   # 原图片这一层的输出
combination_features = layer_features[2, :, :, :]   # 目标图片这一层的输出
loss += content_weight * content_loss(target_image_features, combination_features)
# 计算风格损失
for layer_name in style_layers:
    layer_features = output_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]   # 风格图片当前层的输出
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl    # 这里的损失跟层的深度还有关系，越深的层损失权重越小
# 计算变差损失
loss += total_variation_weight * total_variation_loss(combination_image)


# 针对损失对输入进行求梯度
grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])  # 用于获取当前梯度损失值和当前梯度值的函数


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x):
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values


evaluator = Evaluator()

result_prefix = 'style_transfere_result'
iterations = 20  # 迭代20次

x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print("开始第{}次迭代".format(i))
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print("目前的损失:", min_val)

    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png'%i
    imsave(fname, img)













