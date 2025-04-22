# Face Recognition with CBAM and ArcFace

This project implements a deep learning model for face classification using Convolutional Neural Networks (CNN), CBAM (Convolutional Block Attention Module), and ArcFace for face classification. The model achieves 100% accuracy on the given dataset for classifying faces.



## Dataset

The dataset used in this project is the **Face Identification Dataset**. It contains images of 487 different individuals and is sourced from Kaggle.

- **Dataset Link**: [Face Identification Dataset by Aleksei Zagorskii on Kaggle](https://www.kaggle.com/datasets/juice0lover/face-identification)
- **Access**: The dataset is loaded using KaggleHub for easy access:  
  ```python
  import kagglehub
  data_path = kagglehub.dataset_download('juice0lover/face-identification')


---

## Model Components

### 1. Convolutional Neural Network (CNN)
A CNN is employed for extracting features from input images. It learns hierarchical patterns and representations from the dataset that are crucial for face recognition.

### 2. CBAM (Convolutional Block Attention Module)
The CBAM module is used to refine feature maps through channel and spatial attention. This enhances the model's ability to focus on important regions and channels of the input data.

- **CBAM Implementation**: The implementation is taken from the GitHub repository by **Jongchan**.
  - **Link**: [CBAM GitHub](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)
  - **Paper**: ["CBAM: Convolutional Block Attention Module"](https://arxiv.org/abs/1807.06521)

### 3. ArcFace
The ArcFace function is used to compute the angular margin between class features, providing more discriminative feature representations for face recognition. The implementation of ArcFace is adapted from a Kaggle notebook by **parthdhameliya77**.

- **ArcFace Implementation**: The implementation is taken from a Kaggle notebook by **parthdhameliya77**.
  - **Link**: [ArcFace Kaggle Notebook](https://www.kaggle.com/code/parthdhameliya77/simple-arcface-implementation-on-mnist-dataset)


## Requirements

- Python 3.x
- PyTorch
- KaggleHub
- NumPy
- Matplotlib (for visualization)
- Sklearn
- Opencv
- PIL



---
## Results

The model achieves 100% accuracy on the dataset for classifying faces.


## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


## Acknowledgments

- The dataset used in this project is provided by **Aleksei Zagorskii** and is available on Kaggle:  
  [Face Identification Dataset on Kaggle](https://www.kaggle.com/datasets/juice0lover/face-identification)
  
- The CBAM module is based on the implementation by **Jongchan**.  
  [CBAM GitHub](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)  
  [CBAM Paper](https://arxiv.org/abs/1807.06521)

- The ArcFace implementation is adapted from a Kaggle notebook by **parthdhameliya77**.  
  [ArcFace Kaggle Notebook](https://www.kaggle.com/code/parthdhameliya77/simple-arcface-implementation-on-mnist-dataset)

