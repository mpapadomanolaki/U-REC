# U-REC
Joint optimization of semantic segmentation and image reconstruction

UNet and UREC networks were used to produce the accuracy results of the paper 'Maria Papadomanolaki, Konstantinos Karantzalos, Maria Vakalopoulou. A MULTI-TASK DEEP LEARNING FRAMEWORK COUPLING SEMANTIC SEGMENTATION AND IMAGE RECONSTRUCTION FOR VERY HIGH RESOLUTION IMAGERY. IGARSS , Jul 2019, Yokohama, Japan' (https://hal.inria.fr/hal-02266085/document)

make_xys.py is used to create training and validation csv files with x and y coordinate locations of the images, in order to extract patches during the training process.

custom.py is the definition of the custom dataloader thet we used for the isprs dataset.

tools.py involves functions that are called in m ain.py during training.

infer.py is used for evaluating the model on the testing images.

If you find this code useful in your research, please consider citing:

Maria Papadomanolaki, Konstantinos Karantzalos, Maria Vakalopoulou. A MULTI-TASK DEEP LEARNING FRAMEWORK COUPLING SEMANTIC SEGMENTATION AND IMAGE RECONSTRUCTION FOR VERY HIGH RESOLUTION IMAGERY


