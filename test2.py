import torch
import torch.nn as nn
import numpy as np
import random

_batch_input1 = torch.tensor(np.linspace(1, 12,12).reshape((2,2,3)))
_batch_input2 = torch.tensor(np.linspace(12,23,48).reshape((4,4,3)))
# print(_batch_input1)
# print(_batch_input2)
# _batch_input2[1.1:3,1:3,:] = 0
# print(_batch_input2)
# a = torch.cat((_batch_input2, _batch_input1), axis=1)
# print(a.size())

# a = torch.mean(_batch_input1, dim=2)
# print(a)
# print(a.size())
#
# b = torch.mean(_batch_input1, dim=3)
# print(b)
# print(b.size())
# print(torch.unsqueeze(_batch_input1,dim=-1).size())
# print()
# print(torch.unsqueeze(_batch_input1,dim=3).size())
# print(_batch_input1.transpose(1,0).size())

# input_x = torch.mean(_batch_input1, dim=3, keepdim=True)  # 因为求H的话即是求W上的平均值，
# input_y = torch.mean(_batch_input2, dim=2, keepdim=True)  # 因为求W的话即是求H上的平均值,mean根据dim维度去求那个维度平均值
# print(input_x.size())
# print(input_y.size())
# input_y = input_y.transpose(2, 3)
# print(input_y.size())
# input_com = torch.cat((input_x, input_y), dim=2)
# print(input_com.size())
# intput1_x, intput1_y = input_com.split((input_x.size()[2], input_y.size()[2]), dim=2)
# intput1_y = intput1_y.transpose(2, 3)
# print(intput1_x.size())
# print(intput1_y.size())

# a = nn.AdaptiveAvgPool2d(2)
# print(a(_batch_input2).size())

# a = ["2","333","23","333","45"]
# print(random.sample(a,2))

import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
# A = B = C = np.ones((10, 10), dtype=np.uint8)
# Apoly = Bpoly = Cpoly = PolygonsOnImage(
#     [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
#     shape=(10, 10))
# seq = iaa.Fliplr(0.5)
# print("apoly：", Apoly)
# seq_det = seq.to_deterministic()
# imgs_aug = seq_det.augment_images([A, B, C])
# polys_aug = seq_det.augment_polygons([Apoly, Bpoly, Cpoly])
# print(polys_aug)

# import numpy as np
# from imgaug.augmentables.polys import Polygon, PolygonsOnImage
# import imgaug
# image = np.zeros((100, 100))
# seq = iaa.Fliplr(0.5)
# seq_det = seq.to_deterministic()
# polys = [
#     Polygon([(0.5, 0.5), (100.5, 0.5), (100.5, 100.5), (0.5, 100.5)]),
#     Polygon([(50.5, 0.5), (100.5, 50.5), (50.5, 100.5), (0.5, 50.5)])
# ]
# polys_oi = imgaug.PolygonsOnImage(polys, shape=image.shape)
# polys_aug = seq_det.augment_polygons(polys_oi)
# polys_aug0 = seq_det.augment_polygons(polys_oi)[0]
#
# print(polys_aug)
# print(polys_aug0)

import torch
import numpy as np
from mmengine.structures import InstanceData
from mmocr.data import TextDetDataSample
 # gt_instances
data_sample = TextDetDataSample()
img_meta = dict(img_shape=(800, 1196, 3),pad_shape = (800, 1216, 3))
gt_instances = InstanceData(metainfo=img_meta)
gt_instances.bboxes = torch.rand((5, 4))
gt_instances.labels = torch.rand((5,))
data_sample.gt_instances = gt_instances
assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
len(data_sample.gt_instances)
print(data_sample)