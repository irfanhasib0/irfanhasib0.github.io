

Table of Contenct
==================
### Notes :
__In this document for every project I have shown the flowchart and Result with a brief description. 
For Detail Presentation and Code please go to the Notebook link Provided for every project!!__
- (*) Means only organized code with comments
- (**) Means Code with basic doccumentation.
- (***) Means Code with Good Doccumentation.

### Kaggle Competetions and Job Entrance Problem :
 - [House Price Prediction :: Data Pre-Processing, ANN with tensorflow low level API and and Hiper-Parameter Tuning.  (***)](#house-price-prediction--data-pre-processing-and-hiper-parameter-tuning)
 - [Japanese Job Entrance Problem :: Shakura Bloom Prediction (***)](#japanese-job-entrance-problem--shakura-bloom-prediction)

### Machine Learning Algorithms from Scratch :

 - [Neural Network         :: Implementation from scratch with raw python (*)](#neural-network--nn-implementation-from-scratch)
 - [Decision Tree(ID3)     :: Implementation from scratch with continuous feature support. (***)](#decision-tree--id3-implementation-from-scratch)
 - [Naive Bayes            :: Implementation for text classification with text preprocesing from scratch (**)](#naive-bayes--implementation-for-text-classification)

Neural Network :: NN Implementation from scratch
================================================
### Notebook : [NN Implementation from scratch - Notebook (Project Presentation and Code Link) ](https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/ANN_From_Scratch_modular_class.ipynb)

* Forward Propagation
* Backward Propagation


	a. Forward Propagation b. Back Propagation (Open Image in new tab for full resolution)

<img src="docs/Algorihms/NN_fp.jpg" align="left"
     title="Schematics" width="800" height="480"/>
<img src="docs/Algorihms/NN_bp.jpg" align="center"
     title="(Open Image in new tab for good resolution)" width="800" height="480">

c. Result of ANN implementation for XOR data - mean sqaure error vs epoch -

<img src="docs/Results/xor_ann.jpg" align="center"
     title="(Open Image in new tab for good resolution)" width="320" height="240">
.

Decision Tree :: ID3 Implementation from scratch
====================================================
### Notebook : [ID3 Implementation from scratch - Notebook (Project Presentation and Code Link) ](https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/ID3_with_continuous_feature_support_exp.py)

* __Dataset     :__ Titanic and irish dataset was used for testing ID3.
* __Steps       :__
                  -Continuous data spliting based on information gain.
                  -Information Gain Calculation Function.
                  -ID3 Algorithm according to flowchart.


* __Tuning     :__ Reduced Error Pruining.
* __Prediction :__ Accuracy,Precision,Recall Reporting.
 
 ### ID3 Flow Chart of Implementation (Open Image in new tab for full resolution)

<img src="docs/Algorihms/ID3.jpg" align="center"
     title="(Open Image in new tab for full resolution)
" width="480" height="480">

 #### Result of ID3 implementation for iris data - a.Precision Recall and Accuracy and b.True Label vs Prediction -


<img src="docs/Results/iris_ID3.png" align="left"
     title="(Open Image in new tab for good resolution)
" width="320" height="240">

<img src="docs/Results/irispred.jpg" align="center"
     title="(Open Image in new tab for good resolution)
" width="480" height="240">



 #### Result of ID3 implementation for Titanic data - a.Precision Recall and Accuracy and b.True Label vs Prediction

<img src="docs/Results/titanic_ID3.png" align="left"
     title="(Open Image in new tab for full resolution)
" width="320" height="240">

<img src="docs/Results/titanicpred.jpg" align="center"
     title="(Open Image in new tab for full resolution)
" width="480" height="240">

.

Naive Bayes :: Implementation for text classification
==========================================================

### Notebook : [Naive Bayes Implementation from scratch - Notebook (Project Presentation and Code Link) ](https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/Naive_Bayes_Stack_Exchange.ipynb)

* Archived data from stack exchange is used for classification.
* Text Preprocessing was done with raw python without nltk.
	Naive Bayes algorithm applied on the procesed text.
<img src="docs/Algorihms/Naive_Bayes.jpg" align="center"
     title="Open Image in new tab for good resolution" width="640" height="480">

* Result of Naive Bayes implementation for Stack Exchange data - a.Precision Recall and Accuracy and b.True Label vs Prediction

<img src="docs/Results/stack_exchange_NB.png" align="left"
     title="(Open Image in new tab for full resolution)
" width="320" height="240">

<img src="docs/Results/stack_exchange_NB_pred.png" align="center"
     title="(Open Image in new tab for full resolution)
" width="640" height="240">

House Price Prediction :: Data Pre-Processing and Hiper-Parameter Tuning
==================================================================================================
### Notebook : [House Price Prediction - Notebook (Project Presentation and Code Link) ](https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/ANN_Tensorflow__Kaggle_Houseprice_prediction.ipynb)
   
 *  __Dataset :__ House Price Dataset of kaggle.com
 *  __Steps :__  
                 - Data Preprocessing
                 - Implementing Nural Network with tensorflow low level API.
                 - Hiper-Parameter Tuning for ANN
### 1.0 Project Flow Chart :
<img src="docs/Algorihms/kaggle_hp.jpg" align="center"
     title="Open Image in new tab for good resolution" width="800" height="1000">

### Cross validation(MSLE) and Kaggle Result(RMSLE)
<img src="docs/ANN_HP/cv.png" align="left"
     title="(Open Image in new tab for full resolution)" width="320" height="240"/>
<img src="docs/ANN_HP/res.png" align="center"
     title="(Open Image in new tab for full resolution)" width="480" height="320"/></br>
     

 

Japanese Job Entrance Problem :: Shakura Bloom Prediction
==================================================================================================

### Notebook : [Shakura Bloom Prediction - Notebook (Project Presentation and Code Link)](https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/Sakura_TF_NN_Report.ipynb)
   
 *  __Dataset :__ Weather data from japanese meteorological agency
 *  __Steps :__  
                 - Data Preprocessing
                 - Implementing Nural Network with tensorflow low level API.
                 - Hiper-Parameter Tuning for ANN

 
### 1.0 Project Flow Chart :
<img src="docs/Algorihms/sakura_jp.jpg" align="center"
     title="Open Image in new tab for good resolution" width="640" height="640">
     
### Cross Validation R2 Score :
<img src="docs/Sakura/res.png" align="left"
     title="(Open Image in new tab for full resolution)" width="240" height="160"/></br>


