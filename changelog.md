# Change Log

## 14 Mar, Minor Revision

Changed the logic of b-tree.

## Feb 27, Added Feature

* Added 2-dimensional data generator.

## Feb 20, Major Revision

* Update all models and remove dependencies for PyTorch. Now PyTorch is only used for testing and comparing purposes. Files started with ```pt_``` requires PyTorch.
* Neural Network features, such as the fully connected layer, are provided by [TinyML](https://pypi.org/project/tinyml/). It is Xiaozhe's Master Basic Module and shall be allowed to use (will confirm).
* All models, including the PyTorch one, handles the type conversion, normalization, etc inside.
* Add LIF code, which is still at a very early stage. Since it is not necessary in this project, it can be safely ignored.
* Now RMI model, which is also called staged model, supports linear regression, quadratic regression, general polynomial regression with degree $k$, fully connected neural networks and B-Tree.