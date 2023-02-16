# **CC MADE EASY : Crowd Counting becomes much simpler**

## **Introduction**
This is a dynamic Crowd counting framework implemented with PyTorch, inspired from the [C^3 FrameWork](https://github.com/gjy3035/C-3-Framework) with more features and improvements. We aim to contribute at the crowd counting community by providing an enhanced and more flexible  framework.
Special thanks to [Ihab Had](https://) and [Selmane Fay](https://github.com/goldenfay). If you find this project useful, please **Star** 🌟 this repository to show support.

Any suggestions/contributions or bug reports are welcome.

## **Features**
* **Flexibility and modularity**: The framework offers the possibility to extend any module (Density map generator, DataSet, DataLoader, CNN-CC model ) by simple implementation of a child class inheriting from the corresponding base class.
* **Standalone**: Any module can be executed separately from it directory without need to run the project from the root directory.
* **Multi-platform**: Unlike most of Crowd counting frmeworks, our framework can work fine on any OS , including **Google Colab platform** without need for paths management that's caused by machine OS (common relative path problems).
* **High dynamic environement training**: We are glad to introduce a super adaptable mechanism for training flexibility to work for both CUDA and CPU environements. So if, for some reason, you partially trained a model using CUDA and you don't dispose it anymore, the train could resumed by migrating all model parameters from CUDA to CPU and vis-versa .

## **Getting Started**

* **Prerequisites**
    * Python 3.x
    * Pytorch 1.0 (some networks only support 0.4): http://pytorch.org .
    * Additional libs in requirements.txt, run `pip install -r requirements.txt`.
    
* **Training**

run `process/main.py` with mentionned params to launch all the train process, from GTD generation to train performance evaluation. If any step is already done, the script will automatically skip that phase and pass the to next task.

* **Testing**

The test phase can be executed automatically after the train is done within the script, or launched by passing `--test` argument, with eventualy plot train and test graphs.
