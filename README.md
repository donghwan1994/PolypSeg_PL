# Polyp Segmentation (Pytorch Lightning)

Pytorch Lightning implementation of polyp segementation methods for easy and useful research.

## Getting Started

### Prerequisites

* torch
* torchvision
* pytorch-lightning
* tqdm
* easydict
* pyyaml
* opencv-python
* thop
* tabulate
* scipy

Easy install script : `pip install -r requirements.txt`

### Download dataset

The training and testing datasets come from [PraNet](https://github.com/DengPingFan/PraNet). Download these datasets and unzip them into your data folder.

- [Training Dataset](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)
- [Testing Dataset](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing)

### Training Polypsegmentation methods

Novel polyp segmentation methods ([PraNet](https://github.com/DengPingFan/PraNet), [SANet](https://github.com/weijun88/SANet), [MSNet](https://github.com/Xiaoqi-Zhao-DLUT/MSNet)) are implemented.
You can train them with the below commands.
(For details of the methods, please read the papers and official repositories.)

* PraNet (MICCAI 2020) [[paper](https://arxiv.org/pdf/2006.11392.pdf)][[github](https://github.com/DengPingFan/PraNet)]
```   
python train.py --method 'pranet' --gpus 1 --num_workers 4 --data_root DATA_PATH
```
* SANet (MICCAI 2021) [[paper](https://arxiv.org/pdf/2108.00882.pdf)][[github](https://github.com/weijun88/SANet)]
```   
python train.py --method 'sanet' --gpus 1 --num_workers 4 --data_root DATA_PATH
```
* MSNet (MICCAI 2021) [[paper](https://arxiv.org/pdf/2108.05082.pdf)][[github](https://github.com/Xiaoqi-Zhao-DLUT/MSNet)]
```   
python train.py --method 'msnet' --gpus 1 --num_workers 4 --data_root DATA_PATH
```

Thanks for great works.

## Inference & Evaluation

You can generate and evaluate predictions of trained model by
```
python test.py  --method 'pranet' --gpus 1 --num_workers 4 --data_root DATA_PATH
```

You can also evaluate predictions generated by
```
python eval.py  --data_root DATA_PATH --pred_root RESULT_PATH 
```

## References

* [Pytorch](https://pytorch.org/)
* [Pytorch-Lightning](https://www.pytorchlightning.ai/)
* [PraNet](https://github.com/DengPingFan/PraNet)
* [SANet](https://github.com/weijun88/SANet)
* [MSNet](https://github.com/Xiaoqi-Zhao-DLUT/MSNet)

## Author

Donghwan Hwang