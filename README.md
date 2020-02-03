

Table of Content
===============================

##### Note :
__In this document for projects are shown as :__
   
   - Notebook with implementation code and detail description.
   - Flowchart of the project.
   - Result vedio / gif / graphs.
   - The no of star after every project means the level of documentation.
   
__For detail description and Code please go to the Notebook link Provided for every project!!__
   

### Kaggle Competetions and Job Entrance Problem :
 - [House Price Prediction :: Data Pre-Processing, ANN with tensorflow low level API and and Hiper-Parameter Tuning.  (****)](#house-price-prediction--data-pre-processing-and-hiper-parameter-tuning)
 - [Japanese Job Entrance Problem :: Shakura Bloom Prediction (***)](#japanese-job-entrance-problem--shakura-bloom-prediction)

### Machine Learning Algorithms from Scratch :

 - [Neural Network         :: Implementation from scratch with raw python (*)](#neural-network--nn-implementation-from-scratch)
 - [Decision Tree(ID3)     :: Implementation from scratch with continuous feature support. (****)](#decision-tree--id3-implementation-from-scratch)
 - [Naive Bayes            :: Implementation for text classification with text preprocesing from scratch (**)](#naive-bayes--implementation-for-text-classification)

### Reinforcement Learning ALgorithms from scratch :
 - [DQN(Deep Q Learning) from scratch with Tensorflow-KERAS(**)](#dqn-and-ddpg--implementation-from-scratch)
 - [DDPG(Deep Deterministic Policy Gradient) from scratch with Tensorflow(**)](#dqn-and-ddpg--implementation-from-scratch)
 
### Control Algorithms Implementation from scratch :
- [ILQR(Iterative Linear Quadratic Regulator) Implementation from scratch(****)](#ilqr-and-mpc-implementation-from-scratch-for-self-driving-car-simulator)
- [MPC(Model Predictive Controller) Implementation from scratch(**)](#ilqr-and-mpc-implementation-from-scratch-for-self-driving-car-simulator)

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
                 - Feature Analysis and Data Preprocessing
                 - Implementing Nural Network with tensorflow low level API.
                 - Hiper-Parameter Tuning for ANN
 
### 1.0 Project Flow Chart :
<img src="docs/Algorihms/sakura_jp.jpg" align="center"
     title="Open Image in new tab for good resolution" width="640" height="640">
     
### Cross Validation R2 Score :
<img src="docs/Sakura/res.png" align="center"
     title="(Open Image in new tab for full resolution)" width="240" height="160"/></br>



DQN and DDPG:: Implementation from scratch
==================================================================================================
### Notebook : [Mountain Car with DQN - Notebook (Project Presentation and Code Link)](https://github.com/irfanhasib0/RL-Algorithms/blob/master/Deep_Q_Learning_mc.ipynb)
 
### Notebook : [Pendulum with DDPG - Notebook (Project Presentation and Code Link)](https://github.com/irfanhasib0/RL-Algorithms/blob/master/Deep_Q_Learning_mc.ipynb)

* __DQN Environments :__ OpenAI gym --> Mountain Car ENvironment
* __DDPG Environments :__ OpenAI gym --> Pendulumn Environment
 
                 
### 1.0 Project Flow Chart for DQN and DDPG :

<img src="docs/Algorihms/_DQN.jpg" align="left"
     title="Open Image in new tab for good resolution" width="460" height="400">
     

<img src="docs/Algorihms/_DDPG.jpg" align="center"
     title="Open Image in new tab for good resolution" width="460" height="400"></br>
     

### Results DQN :
<img src="docs/Results/ddpg_rewards.png" align="left"
     title="(Open Image in new tab for full resolution)" width="320" height="240"/></br>
<img src="docs/Results/ddpg_pendulum_.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="320" height="240"/></br>
     
     
### Results DDPG :
<img src="docs/Results/ddpg_rewards.png" align="left"
     title="(Open Image in new tab for full resolution)" width="320" height="240"/></br>
<img src="docs/Results/ddpg_pendulum_.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="320" height="240"/></br>
     
     



ILQR and MPC :: Implementation from scratch for self driving car simulator
==================================================================================================

### Notebook : [ILQR Implementation - Notebook (Project Presentation and Code Link)](https://github.com/irfanhasib0/RL-Algorithms/blob/master/DDPG_Pendulum_TF-V-2-ROS.ipynb)
   
 *  __Simulation Environment :__ AIRSIM a Car Simulator by Microsoft , OpenAI gym Car ENnvironment.
 *  __I0 :__ Input --> Map Points , Output --> Steering Angle, Acclelation, Brake
 *  __Steps :__  
                 
                 - Map Tracker Module
                 -- Input : Takes Map points as Input
                 -- Trajectory Queue : Gets N next points from map points to follow according to car position and orientation.
                 -- Update Trajectory Queue : Update the queue of points to follow with moving car position and orientation.
                 -- Output : Next N points to follow from Trajectory Queue
                 
                 - Data Preprcessor Module : 
                 -- Input : Trajectory points wrt Car position from Map tracker module.
                 -- Full State Calculation : Calculates yaw from ref points and target velocity.
                 -- Relative Trajectory calculation : Relative Trajectory calculation as Car position nad yaw as                         origin.
                 -- Output : Adjusts no of points to track in the refference trajectory accrding to current velocity.  
                 
                 - ILQR Module
                 -- Input : Refferance trajectory to follow from Data Processor module.
                 -- Output : Calculate Optimal steering angle and accelaration with ILQR algorithm shown below.
                 
                 OR,
                 
                 - MPC Module
                 -- Input : Refferance trajectory to follow from Data Processor module.
                 -- Output : Calculate Optimal steering angle and accelaration with MPC algorithm shown below.
          
### 1.0 Project Flow Chart :
<img src="docs/Algorihms/MPC.jpg" align="center"
     title="Open Image in new tab for good resolution" width="640" height="640">
<img src="docs/Algorihms/iLQR.jpg" align="center"
     title="Open Image in new tab for good resolution" width="480" height="480">
     
#### Results (ILQR) :
* OpenAI Gym Car Environment 
* Airsim City Space Environment 
* Airsim Neighbourhood Environment 

<img src="docs/Results/rec_car_env.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="400" height="240"/>
<img src="docs/Results/fig_car_env.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="400" height="240"/>
<img src="docs/Results/airsim_cs.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="400" height="240"/>
<img src="docs/Results/fig_cs.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="400" height="240"/>
<img src="docs/Results/airsim_nh.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="400" height="240"/>
<img src="docs/Results/fig_nh.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="400" height="240"/>

    
     
     




```python
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='img1'></td><td><img src='img2'></td></tr></table>"))

```


<table><tr><td><img src='img1'></td><td><img src='img2'></td></tr></table>



```python
<embed src="file_name.pdf" width="800px" height="2100px" />
```
