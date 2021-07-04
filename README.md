# 3DFCNwithIOUlossforCropMapping
Keras code for our paper "3D Fully Convolutional Neural Networks with Intersection Over Union Loss for Crop Mapping from Multi-Temporal Satellite Images"

Our paper is accepted to IGARSS 2021 and  can be found at: [arXiv](https://arxiv.org/abs/2102.07280).

### Requirements
- [numpy 1.18.5](https://numpy.org/)
- [tensorflow 2.3.1](https://www.tensorflow.org/)
- [tables 3.6.1](https://www.pytables.org/)


If you want to train the model using first four folds, first download the preprocessed data from [GoogleDrive](https://drive.google.com/file/d/1eql-2OsG9mr8fOUi3SMi19HELzzVbbCj/view?usp=sharing), and then run:

```
python train.py --data_dir 'data' --save_dir 'save' --loss_function 'IOU' --validation_fold 5
```

In addition to *data_dir*, *save_dir*, loss_function, and validation_fold, you can set these training configurations: *batch_size, learning_rate, epochs.*

## Citation
```
@article{mohammadi20213d,
  title={3D Fully Convolutional Neural Networks with Intersection Over Union Loss for Crop Mapping from Multi-Temporal Satellite Images},
  author={Mohammadi, Sina and Belgiu, Mariana and Stein, Alfred},
  journal={arXiv preprint arXiv:2102.07280},
  year={2021}
}
```
