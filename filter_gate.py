# -*- coding: UTF-8 -*-
'''
Image encoder.  Get visual embeddings of images.
'''
import os
import imagehash
from PIL import Image
import pickle
class FilterGate():
    def __init__(self,base_path,hash_size):
        self.base_path=base_path
        self.hash_size=hash_size
        self.best_imgs={}

    def phash_sim(self,img1,img2,hash_size=None):
        if not hash_size:
            hash_size=self.hash_size
        img1_hash = imagehash.phash(Image.open(img1), hash_size=hash_size)
        img2_hash = imagehash.phash(Image.open(img2), hash_size=hash_size)

        return 1 - (img1_hash - img2_hash) / len(img1_hash) ** 2


    def filter(self):
        self.best_imgs={}
        ents = os.listdir(self.base_path)
        while len(ents)>0:
            ent=ents.pop()
            imgs=os.listdir(self.base_path + ent + '/')
            n_img=len(imgs)
            sim_matrix=[[0]*n_img for i in range(n_img)]
            for i in range(n_img):
                for j in range(i+1,n_img):
                    sim=self.phash_sim(self.base_path + ent + '/'+imgs[i], self.base_path + ent + '/'+imgs[j])
                    sim_matrix[i][j]=sim
                    sim_matrix[j][i] =sim
            # sim_matrix=[sum(i) for i in sim_matrix]
            max_index=0
            max_sim=sum(sim_matrix[0])
            for i in range(1,n_img):
                if sum(sim_matrix[i])>max_sim:
                    max_index=i
                    max_sim=sum(sim_matrix[i])
            self.best_imgs[ent]=self.base_path + ent + '/'+imgs[max_index]
        return self.best_imgs
#dhash
#     def save_best_imgs(self,output_file,n=1):
#         with open(output_file, 'wb') as out:
#             pickle.dump(self.best_imgs, out)
#     def dHash(image):
#     image_new=image
#     #计算均值
#     avreage = np.mean(image_new) 
#     hash=[]
#     #每行前一个像素大于后一个像素为1，相反为0，生成哈希
#     for i in range(8):
#         for j in range(8):
#             if image[i,j]>image[i,j+1]:
#                 hash.append(1)
#             else:
#                 hash.append(0)
#     return hash

# #计算汉明距离
#     def Hamming_distance(hash1,hash2): 
#         num = 0
#         for index in range(len(hash1)): 
#             if hash1[index] != hash2[index]: 
#                 num += 1
#         return num

# if __name__ == "__main__":
#     image1 = Image.open('image1.png')
#     image2 = Image.open('image2.png')
#     #缩小尺寸并灰度化
#     start = time.time()
#     image1=np.array(image1.resize((9, 8), Image.ANTIALIAS).convert('L'), 'f')
#     image2=np.array(image2.resize((9, 8), Image.ANTIALIAS).convert('L'), 'f')
#     hash1 = dHash(image1)
#     hash2 = dHash(image2)
#     dist = Hamming_distance(hash1, hash2)
#     end = time.time()
#     #将距离转化为相似度
#     similarity = 1 - dist * 1.0 / 64 
#     print('dist is '+'%d' % dist)
#     print('similarity is ' +'%d' % similarity)
#     print('time is  '+ '%f'%(end-start))



if __name__ == '__main__':
    f=FilterGate('./dataset/wn18-images',hash_size=16)
    f.filter()
    f.save_best_imgs('./dataset/wn18_best_img.pickle')

