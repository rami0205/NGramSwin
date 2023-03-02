# NGramSwin

## N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution (CVPR 2023)
Haram Choi<sup>*</sup>, Jeongmin Lee, and Jihoon Yang<sup>+</sup>

<sup>*</sup>: This work has been done during my 3rd semester of Master Course in Sogang University.

<sup>+</sup>: Corresponding author.

- Introduces the N-Gram context to deep learning in the low-level vision domain.
- Our N-Gram context algorithm used at window partitioning is in ```my_model/ngswin_model/win_partition.py```
- Two tracks of this paper: 
1. Constructing NGswin with an efficient architecture for image super-resolution.
2. Improving other Swin Transformer based SR methods (SwinIR-light, HNCT) with N-Gram.
- NGswin outperforms the previous leading efficient SR methods with a relatively efficient structure.
- SwinIR-NG outperforms the current best state-of-the-art lightweight SR methods.

### News
-**Mar 02, 2023:** Codes released publicly

-**Feb 28, 2023:** Our paper accepted at CVPR 2023

### Model Architecture
![model_architecture](https://user-images.githubusercontent.com/69415453/222345346-85128b90-00f9-4be6-8c22-d96f250bd9ce.png)

### Visual Results

<details>
<summary>Comparison with other models (Please Click)</summary>

![vis_results](https://user-images.githubusercontent.com/69415453/222345740-3683332b-2564-420d-8d88-2ed70d87aee6.png)

</details>

<details>
<summary>Visualization of effectiveness of N-Gram context (Please Click)</summary>

![vis_results2](https://user-images.githubusercontent.com/69415453/222345749-ec725538-76be-483a-85cd-b025b89d976c.png)

</details>

#### * The visual results on the other images can be downloaded in my [drive](https://1drv.ms/u/s!AoUesdU_BVZrirpfrVniRoBRqjSw2Q?e=dh2ucT).
- The visual results can be produced by running the codes below.

- The datasets (Set5, Set14, BSDS100, Urban100, Manga109) are publicly released and low-resolution images can be obtained by MATLAB bicubic kernel (codes in ```bicubic_kernel.m```).


### Efficient and Lightweight Super-Resolution Results

<details>
<summary>NGswin Efficient SR Results (Please Click)</summary>

![NGswin_results](https://user-images.githubusercontent.com/69415453/201585668-a5ca0e65-77d9-4648-8199-62e277218a7b.png)

</details>

<details>
<summary>SwinIR-NG Lightweight SR Results (Please Click)</summary>

![comp_sota2](https://user-images.githubusercontent.com/69415453/222380816-ecc4a354-5d59-4692-9c7e-8cbedb8ab9c5.png)

</details>

<details>
<summary>Summary of Results (Please Click)</summary>

![github_result](https://user-images.githubusercontent.com/69415453/202172117-804639b1-c370-4858-a6e8-16ad3ba5ba47.png)

</details>

## Requirements

### Libraries
* Python 3.6.9
* PyTorch >= 1.10.1+cu102
* timm >= 0.6.1
* torchvision >= 0.11.2+cu102
* einops 0.3.0
* numpy 1.19.5
* OpenCV 4.6.0
* tqdm 4.61.2
* (optional) MATLAB (for BICUBIC kernel to obtain low-resolution images)


### Datasets (names and path)

```TESTING```
```
HR file name example: baby.npy
LR file name example: babyx2.npy
```
> ../testsets
> > Set5
> > > HR
> > > 
> > > LR_bicubic
> > > > X2
> > > > 
> > > > X3
> > > > 
> > > > X4
> > 
> > Set14
> > 
> > BSDS100
> > 
> > urban100
> > 
> > manga109

```TRAINING```
> ../DIV2K
> > DIV2K_train_HR
> > 
> > DIV2K_train_LR_bicubic
> > > X2
> > > 
> > > X3
> > > 
> > > X4

## Testing with pre-trained models
### You can get the results in the Tables of our paper.

If you have multi gpus that can be used for Distributed Data Parallel (DDP), follow the commands below.

Please properly edit the first five arguments to work on your devices.
```
python3 ddp_test_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name NGswin --target_mode light_x2
python3 ddp_test_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name NGswin --target_mode light_x3
python3 ddp_test_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name NGswin --target_mode light_x4
python3 ddp_test_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name SwinIR-NG --target_mode light_x2
python3 ddp_test_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name SwinIR-NG --target_mode light_x3
python3 ddp_test_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name SwinIR-NG --target_mode light_x4
```

If not, follow the commands below.

Please properly edit the first two arguments to work on your devices.
```
python3 dp_test_main.py --device cuda:0 --num_device 1 --model_name NGswin --target_mode light_x2
python3 dp_test_main.py --device cuda:0 --num_device 1 --model_name NGswin --target_mode light_x3
python3 dp_test_main.py --device cuda:0 --num_device 1 --model_name NGswin --target_mode light_x4
python3 dp_test_main.py --device cuda:0 --num_device 1 --model_name SwinIR-NG --target_mode light_x2
python3 dp_test_main.py --device cuda:0 --num_device 1 --model_name SwinIR-NG --target_mode light_x3
python3 dp_test_main.py --device cuda:0 --num_device 1 --model_name SwinIR-NG --target_mode light_x4
```

## Training from scratch: x2 task
### with DDP
- NOTE: argument ```batch_size``` means the size of mini-batch assigned per gpu
```
python3 ddp_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name NGswin --batch_size 16 --target_mode light_x2
python3 ddp_main.py --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name SwinIR-NG --batch_size 16 --target_mode light_x2
```

### without DDP
- NOTE: argument ```batch_size``` means the size of mini-batch assigned to total gpus (differs from DDP)
```
python3 dp_main.py --device cuda:0 --num_device 4 --model_name NGswin --batch_size 64 --target_mode light_x2
python3 dp_main.py --device cuda:0 --num_device 4 --model_name SwinIR-NG --batch_size 64 --target_mode light_x2
```

## Training by warm-start: x3, x4 tasks
### with DDP
```
python3 ddp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name NGswin --batch_size 16 --target_mode light_x3
python3 ddp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name NGswin --batch_size 16 --target_mode light_x4
python3 ddp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name SwinIR-NG --batch_size 16 --target_mode light_x3
python3 ddp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --total_nodes 1 --gpus_per_node 4 --node_rank 0 --ip_address xxx.xxx.xxx.xxx --backend nccl --model_name SwinIR-NG --batch_size 16 --target_mode light_x4
```

### without DDP
```
python3 dp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --device cuda:0 --num_device 4 --model_name NGswin --batch_size 64 --target_mode light_x3
python3 dp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --device cuda:0 --num_device 4 --model_name NGswin --batch_size 64 --target_mode light_x4
python3 dp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --device cuda:0 --num_device 4 --model_name SwinIR-NG --batch_size 64 --target_mode light_x3
python3 dp_main_finetune.py --pretrain_path models/xxxxxxxx_xxxxxx/model_xxx.pth --warm_start True --warm_start_epoch 50 --device cuda:0 --num_device 4 --model_name SwinIR-NG --batch_size 64 --target_mode light_x4
```

### Citation
```
@article{choi2022n,
  title={N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution},
  author={Choi, Haram and Lee, Jeongmin and Yang, Jihoon},
  journal={arXiv preprint arXiv:2211.11436},
  year={2022}
}
```

## Credits
#### Our codes were strongly referred to Swin Transformer (V1 & V2) and SwinIR.
###### SwinV1: https://arxiv.org/abs/2103.14030
###### SwinV2: https://arxiv.org/abs/2111.09883
###### SwinIR: https://arxiv.org/abs/2108.10257
