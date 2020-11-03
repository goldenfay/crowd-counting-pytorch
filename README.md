# **CC MADE EASY : Crowd Counting becomes much simpler**

## **Introduction**
This is a dynamic Crowd counting framework implemented with PyTorch, inspired from the [C^3 FrameWork](https://github.com/gjy3035/C-3-Framework) with more features and improvements. We aim to contribute at the crowd counting community by providing an enhanced and more flexible  framework.
Special thanks to [Ihab Had](https://) [and Selmane Fay](https://).
Any suggestion/contribution or bug reports are welcome.

## **Features**
* **Flexibility and modularity**: The framework offers the possibility to extend any module (Density map generator, DataSet, DataLoader, CNN-CC model ) by simple implementation of a child class inheritting from corresponding base class.
* **Multi-platform**: Unlike most of Crowd counting frmeworks, our framework can work fine on any OS , including **Google Colab platform** without paths managing issues (common relative paths problem).
* **High dynamic environement training**: We are glab to introduce a super flexible mechanism for train flexibility to work for both CUDA and CPU environements. So if, for some reason, you partially trained a model using CUDA and you don't dispose it anymore, the train could resume by migrating all model parameters from CUDA to CPU and vis-versa .

## **Getting Started**

* **Prerequisites**
    * Python 3.x
    * Pytorch 1.0 (some networks only support 0.4): http://pytorch.org .
    * Additional libs in requirements.txt, run `pip install -r requirements.txt`.
    
* **Training**
run `process/main.py` with mentionned params to launch all the train process, from GTD generation to train performance evaluation. If any step is already done, the script will automatically skipp that phase and pass the to next task.

* **Testing**
The test phase can be executed automatically after train is done within the script or launched by passing `--test` argument, with eventualy plot train and test graphs.
