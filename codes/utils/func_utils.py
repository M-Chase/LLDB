import numpy as np

import torch
import numpy as np
from tqdm import trange
from torch import Tensor
import math

CUDA = torch.cuda.is_available()
def use_gt_mean_easy(img_out,img_gt):
    mean_gray_out = cv2.cvtColor(img_out.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    mean_gray_gt = cv2.cvtColor(img_gt.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    normal_img_adjust = np.clip(img_out * (mean_gray_gt / mean_gray_out), 0, 1)

    # normal_img = (normal_img_adjust * 255).astype(np.uint8)
    return normal_img_adjust

def cal_psnr(adjusted_image, target_image, max_pixel_value=1.0):
    # 计算均方误差(MSE)
    # print(adjusted_image.shape,target_image.shape)
    # mse = F.mse_loss(adjusted_image, target_image)
    img1 = adjusted_image.astype(np.float64)
    img2 = target_image.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    # 计算PSNR
    psnr_value = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    
    return psnr_value

def adjust_gamma(image, gamma=1.0):
    max_val =np.max(image)
    image_normalized = image / max_val
    image_gamma_corrected = np.power(image_normalized, gamma)
    image_gamma_corrected = image_gamma_corrected * max_val
    return image_gamma_corrected

def find_optimal_gamma(source_image, target_image, gamma_range, max_pixel_value):
    optimal_gamma = gamma_range[0]
    max_psnr = 0
    for gamma in gamma_range:
        adjusted_image = adjust_gamma(source_image, gamma)
        psnr = cal_psnr(adjusted_image, target_image, max_pixel_value)
        if psnr > max_psnr:
            max_psnr = psnr
            optimal_gamma = gamma
    
    return optimal_gamma

def use_gt_mean_hard(img_out,img_gt):
    gamma_values = np.linspace(0.4, 3.5, 400)
    optimal_gamma = find_optimal_gamma(img_out, img_gt, gamma_values, 1.0)
    print("最佳的optimal gamma是：",optimal_gamma)
    generated_image = adjust_gamma(img_out, optimal_gamma)
    return generated_image

def split_image_np(image_np, x):

    img_height, img_width, channels = image_np.shape
    
    block_height = img_height // x
    block_width = img_width // x
    
    # 使用reshape和transpose来分割图像
    reshaped = image_np.reshape((x, block_height, x, block_width, channels))
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    blocks = transposed.reshape(-1, block_height, block_width, channels)
    
    return list(blocks)

class kmeans_core:
    def __init__(self, k, data_array, batch_size=8e5, epochs=200, all_cuda=True):
        """
        kmeans by batch
        k: number of the starting centroids
        data_array:numpy array of data
        batch_size:batch size
        epochs: max epoch iterations, if the centeroids not shifting any more, the calculation will cease before this max number
        all_cuda: do you want to move the entire array to the cuda
        
        About data loader: We didn't use dataloader. The data loader will load data entry by entry with cpu multi processor, hence losing the power of fast gpu. Matter of fact, when I use the dataloader the 92.1% of the time consumption is caused by data loader
        """
        self.k = k
        self.data_array = data_array
        self.tensor = Tensor(self.data_array,)
        self.all_cuda = all_cuda
        if all_cuda and CUDA:
            self.tensor = self.tensor.cuda()
        
        self.dim = data_array.shape[-1]
        self.data_len = data_array.shape[0]
        
        self.cent = Tensor(data_array[np.random.choice(range(self.data_len), k)])
        
        if CUDA:
            self.cent = self.cent.cuda()
            
        self.epochs = epochs
        self.batch_size = int(batch_size)
        self.iters = math.ceil(self.data_array.shape[0]/self.batch_size)
        self.index = 0
        

    def get_data(self,index):
        return self.tensor[index:index+self.batch_size,...]

    def run(self):
        for e in range(self.epochs):
            # t = trange(self.iters)
            t = self.iters

            start = self.cent.clone()
            for i in range(t):
                dt = self.get_data(self.index)
                self.index += self.batch_size
                
                if CUDA and self.all_cuda==False:
                    dt = dt.cuda()  
                self.step(dt)
                # t.set_description("🔥[epoch:%s\t iter:%s]🔥 \t🔥k:%s\t🔥distance:%.3f" % (e, i, self.k, self.distance))
            self.index=0
            
            if self.cent.size()[0] == start.size()[0]:
                if self.cent.sum().item() == start.sum().item():
                    # print("Centroids is not shifting anymore")
                    break
                    
        # t = trange(self.iters)
        t = range(self.iters)
        
        for i in t:
            dt = self.get_data(self.index)
            self.index += self.batch_size
            if CUDA and self.all_cuda==False:
                dt = dt.cuda()
            if i == 0:
                self.idx = self.calc_idx(dt)
            else:
                self.idx = torch.cat([self.idx, self.calc_idx(dt)], dim=-1)
        self.index=0
        return self.idx

    def step(self, dt):
        idx = self.calc_idx(dt)
        self.new_c(idx, dt)

    def calc_distance(self, dt):
        bs = dt.size()[0]
        distance = torch.pow(self.cent.unsqueeze(0).repeat(bs, 1, 1) - dt.unsqueeze(1).repeat(1, self.k, 1), 2).mean(
            dim=-1)
        return distance

    def calc_idx(self, dt):
        distance = self.calc_distance(dt)
        self.distance = distance.mean().item()
        val, idx = torch.min(distance, dim=-1)
        return idx

    def new_c(self, idx, dt):
        if CUDA:
            z = torch.cuda.FloatTensor(self.k, self.dim).fill_(0)
            o = torch.cuda.FloatTensor(self.k).fill_(0)
            ones = torch.cuda.FloatTensor(dt.size()[0]).fill_(1)
        else:
            z = torch.zeros(self.k, self.dim)
            o = torch.zeros(self.k)
            ones = torch.ones(dt.size()[0])
            
        ct = o.index_add(0, idx, ones)

        # slice to remove empety sum (no more such centroid)
        slice_ = (ct > 0)

        cent_sum = z.index_add(0, idx, dt)[slice_.view(-1, 1).repeat(1,self.dim)].view(-1, self.dim)
        ct = ct[slice_].view(-1, 1)

        self.cent = cent_sum / ct
        self.k = self.cent.size()[0]

class iou_km(kmeans_core):
    def __init__(self, k, data_array, batch_size=1000, epochs=200):
        super(iou_km, self).__init__(k, data_array, batch_size=batch_size, epochs=epochs)

    def calc_distance(self, dt):
        """
        calculation steps here
        dt is the data batch , size = (batch_size , dimension)
        self.cent is the centroid, size = (k,dim)
        the return distance, size = (batch_size , k), from each point's distance to centroids
        """
        bs = dt.size()[0]
        box = dt.unsqueeze(1).repeat(1, self.k, 1)
        anc = self.cent.unsqueeze(0).repeat(bs, 1, 1)

        outer = torch.max(box[..., 2:4], anc[..., 2:4])
        inner = torch.min(box[..., 2:4], anc[..., 2:4])

        inter = inner[..., 0] * inner[..., 1]
        union = outer[..., 0] * outer[..., 1]

        distance = 1 - inter / union

        return distance

