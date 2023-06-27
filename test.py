import numpy as np
import cv2

# lmk = np.array([range(0,10)]).reshape(5,2)
# lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
# print(lmk_tran)
# src = cv2.imread(r"D:\test\3b87e950352ac65c13500793f2f2b21192138afc.jpg")
# M = np.array([[ 1.71473164, -0.185536013, -2239.49061],
#  [ 1.85536013e-01,  2.71473164e+00, -1.01229068e+03]])
# M = np.float64(M)
# cv2.imshow("before", src)
# ndimage = cv2.warpAffine(src, M, (112,112), borderValue=0.0)
# cv2.imshow("test", ndimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
list_path = r"D:\project\test\data\chinese_english_digits.txt"
list_path1 = r"D:\project\test\data\self_dict.txt"
result_path = r"D:\project\test\data\chinese_english_digits1.txt"
a = set()
with open(list_path1, 'r', encoding='utf8') as f1:
 a = set(f1.read())

with open(list_path, 'r', encoding='utf8') as f:
 name_list = f.read()
 # print(name_list)
 ret = set(name_list)
 print(ret.__sizeof__())
 ret = ret.union(a)
 print(ret.__sizeof__())
 ret = list(ret)
 ret.sort()
 # print(ret)
 result = '\n'.join(ret)
 # print(result)
 with open(result_path, 'w', encoding='utf8') as d:
  d.write(result)

print("\u2161")