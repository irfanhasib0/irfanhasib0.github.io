Table of Content
===============================

##### Note :
__In this document for projects are shown as :__
   
   - Link of Jupyter Notebook with implementation code and detail description.
   - Flowchart of the project.
   - Result vedio / gif / graphs.
   - The no of star after every project means the level of documentation.
   
__For detail description and Code please go to the Notebook link Provided for every project!!__
   

### Kaggle Competetions and Job Entrance Problem :
 - [House Price Prediction :: Data Pre-Processing, ANN with tensorflow low level API and and Hiper-Parameter Tuning.  (****)](#house-price-prediction--data-pre-processing-and-hiper-parameter-tuning)
 - [Japanese Job Entrance Problem :: Shakura Bloom Prediction (***)](#japanese-job-entrance-problem--shakura-bloom-prediction)

### Machine Learning Algorithms from Scratch :

 - [Neural Network         :: Implementation from scratch with raw python (**)](#neural-network--nn-implementation-from-scratch)
 - [Decision Tree(ID3)     :: Implementation from scratch with continuous feature support. (****)](#decision-tree--id3-implementation-from-scratch)
 - [Naive Bayes            :: Implementation for text classification with text preprocesing from scratch (**)](#naive-bayes--implementation-for-text-classification)

### Reinforcement Learning ALgorithms from scratch :
 - [DQN(Deep Q Learning) from scratch with Tensorflow-KERAS(**)](#dqn-and-ddpg--implementation-from-scratch)
 - [DDPG(Deep Deterministic Policy Gradient) from scratch with Tensorflow(**)](#dqn-and-ddpg--implementation-from-scratch)
 
### Control Algorithms Implementation from scratch :
- [ILQR(Iterative Linear Quadratic Regulator) Implementation from scratch(****)](#ilqr-and-mpc-implementation-from-scratch-for-self-driving-car-simulator)
- [MPC(Model Predictive Controller) Implementation from scratch(**)](#ilqr-and-mpc-implementation-from-scratch-for-self-driving-car-simulator)

### CNN Projects :
- [Yolo with KERAS]
- [Unet with KERAS]

### ROS Project : (Not well doccumented)

- [ROS : Simple two linked robot inspired from rrbot(-)](ros--simple-two-linked-robot-inspired-from-rrbot)
- [ROS : Writing a script for driving husky robot and getting feed back]

### Embedded System Projects for Pi Labs BD Ltd :
 - Vault Sequirity : IOT based Vault sequitrity System with AVR Microcontroller
 - Safe Box : GPRS based Tracking System with AVR Microcontroller
 - Syringe Pump : RTOS Progmable Infusion Pump with AVR Microcontroller
 - Digital Weight Machine : Server based digital wight Machine with AVR Microcontroller
 - [Link of presentation.](#embedded-system-projects--pi-labs-bd-ltd)

### Academic Project and Thesis (Undergrad) :
 - Remote rescue robot with AVR Microcontroller
 - Car velocity mseasuring system for drive cycle of Dhaka
 - [Link of presentation.](#academic-project-and-thesis)

House Price Prediction :: Data Pre-Processing and Hiper-Parameter Tuning
==================================================================================================
### Notebook : [House Price Prediction - Notebook (Project Presentation and Code Link) ](https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/ANN_Tensorflow__Kaggle_Houseprice_prediction.ipynb)

### Overview :

 *  __Dataset :__ House Price Dataset of kaggle.com
 * __Data is preprocessed :__
        - 2.0 Correlation Analysis
        - 2.1 Outlier handling
        - 2.2 Missing value Handling
        - 2.3 Catagorical to numerical conversion
        - 2.4 Unskewing while needed 
        - 2.5 Data scaling.
 * __ANN Class with tensorflow low level API :__
        - Method train
        - Method predict
        - Method save weights
        - Method load weights
 * Each preprocessing step's effectiveness is checked by simple linear regression.
 * __Hiperparameter Tuning :__
        -4.1 Layer Optimization
        -4.2 Optimizer grid search (Each optimizer for learning rates and epochs)
        -4.3 Learning Rate Optimization
        -4.4 Epochs, Early stopping, Regularization constan optimization.
        -4.5 Activation optimazation
        -4.6 Batch size optimization
 * Cross-Validation with 3 fold was done for overfitting testing.
 * Test result was submitted in kaggle.com for evaluation screenshot can be found at result section.
 * All the graphs of Data preprocessing and Hiperparameter Tuning can be found in [Notebook](https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/ANN_Tensorflow__Kaggle_Houseprice_prediction.ipynb).
 
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
 * __Data Preprocessing :__
    
     -Converting features of a whole year(365/366 samples) to a single vector for adding feature vector.
       --  Converting all the 365 days daya to a single feature
         --- 1. Days after bloom can be ignored. As weather before bloom can effect blooom date.
         --- 2. Days days before Dj- 'Last day of hibernation' is less significant
         --- 3. Feature with high correlation i.e max temp, hr1 preci can be specially considered for processing.
       --  Converting all the 365 days daya to a single feature
         --- 1. Mean of first 90 days
         --- 2. Mean of 30-90 th days
         --- 3. mean of Dj-Dj+45 days
         --- 4. Mean Dj-Dj+60 days
         --- 5. Mean of Dj-Dj+75

     -Feature Selection
         -- Co-relation analysis.
         -- Accuracy of the linear regression after adding each feature.
         
     -Highly Corelated feaure specially anayzing and processing.
     -Outlier analysis.
     -Scaling the data considering interquantile range.

 * __Hiperparameter Tuning :__
        -1 Layer Optimization
        -2 Optimizer grid search (Each optimizer for learning rates and epochs)
        -3 Learning Rate Optimization
        -4 Epochs, Early stopping, Regularization constant optimization.
        
### 1.0 Project Flow Chart :
<img src="docs/Algorihms/sakura_jp.jpg" align="center"
     title="Open Image in new tab for good resolution" width="640" height="640">
     
### Cross Validation R2 Score :
<img src="docs/Sakura/res.png" align="center"
     title="(Open Image in new tab for full resolution)" width="240" height="160"/></br>


Neural Network :: NN Implementation from scratch
================================================
### Notebook : [NN Implementation from scratch - Notebook (Project Presentation and Code Link) ](https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/ANN_From_Scratch_modular_class.ipynb)

### Overview : 
   Here I implemented Neural Network of 3 Layers. I have implemented A Layes Class and functions 
   for Forward propagation,backward propagation and updating weights. I just tested it XOR data.
   It was fitting good.
   
   *  __Forward Propagation :__
       
          -Output of a neuron : Z = W*X + B , Mentioned as Neuron Output
            
            --W(a,b) a for current layer, b for previous layer
            --Layer2:Z1 = Layer2:W11 * X_layer1_node1 + Layer2:W21 * X_layer_node_2 + Layer2:Bias
            --Layer2:Z2 = Layer2:W11 * X_layer1_node1 + Layer2:W21 * X_layer_node_2 + Layer2:Bias
            
          -Adding Non-Linearity : A = Activation(Z) [ Act(Z) in the flow chart] , Mentioned as Activation Output.
          (For detail better under understanding mutual weight indexing W(a,b) see appendix on Nodes)
          
          
   *  __Calulating gradients :__
                     
          -Calculatin dE/dW (dE/dW : grad of error wrt each weights )
            --dE/dW = dE/dA * dA/DZ * dZ/dw 
            
          -Calculatin dE/dA  (dE/dA : grad of error wrt each Activation output )
            --dE/dA : For Last Layer, if loss function is mean square error, then E = 1/2*(Y - A)^2 so , dE/dA = (A-Y)
            --dE/dA : For Other Layers dE/dA will be inherited from the each of the node of next layer acccording
            to mutual weights. 
            
            --Layer2:Error1 = layer3:Error1 * Layer3:W11 + Layer3:Error2 * Layer3:W21 
            
            --Mathmetically Implmented as Error_l2 = Weights_l3.Transpose() * Error_l3
          
          -Calculating dA/dZ (dA/dZ : grad of Activation Output wrt each Neuron output  also mentioned as delta here)
            --It is just Derivative of sigmoid function in my case : delta = A*(1-A)
          
          -Calculating dZ/dW (dZ/dW : grad of Neuron Output wrt each weights )
            --As Z = W*X +B so dZ/dW = X , where for first layer, X is the input 
            --For other layers, the output from previous layer.
       
    
   *  __Updating Weights:__
       
          -Updating each weights W = W + alpha*dW/dE
          -Here alpha is the learning rate.
          
       
   * __Note :__ In the flowchart Gradient calcultion is shown in back propagation.

### Process Flow Chart :

    (Open Image in new tab for full resolution)

<img src="docs/Algorihms/NN_fp.jpg" align="left"
     title="Schematics" width="800" height="480"/>
<img src="docs/Algorihms/NN_bp.jpg" align="center"
     title="(Open Image in new tab for good resolution)" width="800" height="480">

#### Result : 
     Result of ANN implementation for XOR data - mean sqaure error vs epoch -

<img src="docs/Results/xor_ann.jpg" align="center"
     title="(Open Image in new tab for good resolution)" width="320" height="240">

#### Appendix :
* __Nodes :__
         
         - Every Node/Neuron is considered to have weights for each of previous layers node.
         
         ```
         Leyer 1 ->2 nodes 
         Layer 2 ->3 Nodes
         So, Layer 2's each of the 3 Nodes will have two weights for Layer 1's each of the 2 nodes.
         they are -
         Layer 2 - Node 1: W11 , W12
         Layer 2 - Node 2: W21 , W22
         Layer 2 - Node 3: W31 , W32
         Value of Layer 2:Node1 (Z21) = X1 * Layer2:W11 + X2 * Layer2:W12
         Mutual Weights: Layer:W(a,b) : 
             a is the node we are calculating for (current layer)
             b is the contributing node (from previous layer)
             i.e 
              Layer2:W12 --> Layer 2's 1st node's weight for previous layers(Layer 1) 2nd node
              Layer2:W13 --> Layer 2's 1st node's weight for previous layers(Layer 1) 3rd node
         
         ```
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


ILQR and MPC :: Implementation from scratch for self driving car simulator
==================================================================================================

### Notebook : [ILQR Implementation - Notebook (Project Presentation and Code Link)](https://github.com/irfanhasib0/RL-Algorithms/blob/master/DDPG_Pendulum_TF-V-2-ROS.ipynb)


 
### Overview :
 *  __Simulation Environment :__ 
    
   - AIRSIM a Car Simulator by Microsoft 
   - OpenAI gym Car ENnvironment.
    
 * Original Paper of ILQR Part : Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization By - Y Tassa  
 *  __I0 :__ Input --> Map Points , Output --> Steering Angle, Acclelation, Brake
 *  __Steps :__  
                 
                 - Map Tracker Module
                 -- Input : Takes Map points as Input
                 -- Trajectory Queue : Gets N next points from map points to follow according to car position and orientation.
                 -- Update Trajectory Queue : Update the queue of points to follow with moving car position and orientation.
                 -- Output : Next N points to follow from Trajectory Queue
                 
                 - Data Preprcessor Module : 
                 -- Input : Trajectory points from Map tracker module.
                 -- Full State Calculation : Calculates yaw from ref points and target velocity.
                 -- Relative Trajectory calculation : Relative Co-ordinates calculation as Car position nad yaw as                         origin.
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
     title="Open Image in new tab for good resolution" width="700" height="480">
<img src="docs/Algorihms/iLQR_Algorithm_up.jpg" align="center"
     title="Open Image in new tab for good resolution" width="700" height="480">


     
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
#### Appendix : Map Tracker 
<img src="docs/Algorihms/map_tracker.jpg" align="center"
     title="Open Image in new tab for good resolution" width="700" height="480">
     
     




DQN and DDPG:: Implementation from scratch
==============================================================
### Notebook : [Mountain Car with DQN - Notebook (Project Presentation and Code Link)](https://github.com/irfanhasib0/RL-Algorithms/blob/master/Deep_Q_Learning_mc.ipynb)
 
### Notebook : [Pendulum with DDPG - Notebook (Project Presentation and Code Link)](https://github.com/irfanhasib0/RL-Algorithms/blob/master/Deep_Q_Learning_mc.ipynb)

* __DQN Environments :__ OpenAI gym --> Mountain Car ENvironment
* __DDPG Environments :__ OpenAI gym --> Pendulumn Environment
 
                 
### 1.0 Project Flow Chart for DQN and DDPG :

<img src="docs/Algorihms/_DQN.jpg" align="left"
     title="Open Image in new tab for good resolution" width="400" height="320">
     

<img src="docs/Algorihms/_DDPG.jpg" align="center"
     title="Open Image in new tab for good resolution" width="400" height="320"></br>
     

### Results 
* a. Results DQN on Mountain Car (Left):
* b. Results DDPG on Pendulum (Right):
* c. Tset DQN on Mountain Car (Left):
* d. Test DDPG on Pendulum (Right):

<img src="docs/Results/DQN_MC.png" align="left"
     title="(Open Image in new tab for full resolution)" width="240" height="240"/>
<img src="docs/Results/DDPG_PEND.jpg" align="left"
     title="(Open Image in new tab for full resolution)" width="240" height="240"/>
     
<img src="docs/Results/mc.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="240" height="240"/>
<img src="docs/Results/pend.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="240" height="240"/>
     
     




#### ROS : Simple two linked robot inspired from rrbot

- URDF Link(https://github.com/irfanhasib0/ros_ws/tree/master/src/rrbot/rrbot_description)
- Controller Link(https://github.com/irfanhasib0/ros_ws/tree/master/src/rrbot/rrbot_control)
- Gazebo Link(https://github.com/irfanhasib0/ros_ws/tree/master/src/rrbot/rrbot_gazebo)
- Vedio Link (https://youtu.be/lJbyy89X7gM)



Embedded System Projects : Pi Labs BD LTD
=====================================

All these projects I did as an employee of Pi Labs BD Ltd. www.pilabsbd.com
<img src="docs/old/vault_sequrity.jpg" align="left" 
     title="(Open Image in new tab for full resolution)" width="640" height="480"/>
     
<img src="docs/old/safe_box.jpg" align="left"
     title="(Open Image in new tab for full resolution)" width="640" height="480"/>
     
<img src="docs/old/syringe_pump.jpg" align="left"
     title="(Open Image in new tab for full resolution)" width="640" height="480"/>
     
<img src="docs/old/weight_machine.jpg" align="left"
     title="(Open Image in new tab for full resolution)" width="640" height="480"/>




Academic Project and Thesis:
=========================
* My undergrad project of intrumentation and measurement course
* My undergrad thesis

<img src="docs/old/thesis_project.jpg" align="left"
     title="(Open Image in new tab for full resolution)" width="640" height="480"/>



```python

```
