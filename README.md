<a id="table_of_content_link"></a> 
<div class="columns">
<h6> Machine Learning Algorithms from Scratch  </h6>
<ul>
  <li><a style="color:navyblue;font-size:15px;" 
         href ='#ann_link'> Neural Network</a></li>
  <li><a style="color:navyblue;font-size:15px;" 
         href ='#id3_link'> Decision Tree(ID3)</a></li>
  <li><a style="color:navyblue;font-size:15px;" 
         href ='#id3_link'> SVM, Logistic Regression</a></li>
  <li><a style="color:navyblue;font-size:15px;" 
         href ='#naive_bayes_link'> Naive Bayes, KNN</a></li>
</ul>
    
<h6> RL Algorithms from scratch  </h6>
<ul>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#dqn_ddpg_link'> DQN </a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#dqn_ddpg_link'> DDPG </a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#dqn_ddpg_link'> PPO </a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#dqn_ddpg_link'> A2C </a></li>
</ul>

<h6> CNN based ALgorithms with Tensorflow  </h6>
<ul>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#yolo_link'> Yolo-V2.0 : Object Detection </a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#unet_link'> Unet : Semantic Segmentation </a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#flownet_link'> FlowNet : Optical Flow Estimation </a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#disp_link'> Disparity Estimation </a></li>
</ul>
</div> 

<div class="columns">
<h6> Control Algorithms from scratch  </h6>
<ul>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#ilqr_mpc_link'> ILQR </a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#ilqr_mpc_link'> MPC </a></li>
</ul>

<h6> Feature Engineering and Model Tuning  </h6>
<ul class="col1">
  <li><a style="color:navyblue;font-size:15px;" 
href ='#house_price_link'> Kaggle House Price Prediction.</a></li>
  <li><a style="color:navyblue;font-size:15px;" 
         href ='#sakura_link'> Shakura Bloom Prediction.</a></li>
</ul>

<h6> ROS & Robotics </h6>
<ul>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#ros_rrbot_link'> ROS : Simple two linked robot</a></li>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#ros_husky_link'> ROS : Husky robot driver</a></li>
</ul>
</div>

<div class="columns">
<h6> International Robotics Compititions </h6>
<ul>
  <li><a style="color:navyblue;font-size:15px;" 
href ='#urc_2016_link'> University Rover Challenge - 2016</a></li>
    <li>-</li>
    <li>-</li>
    <li>-</li>
    <li>-</li>
</ul>


<h6> Embedded System Projects for Pi Labs BD Ltd </h6>
 <ul>
 <li>-Vault Sequirity System</li>
 <li>-Programmable Syringe Infusion Pump</li>
 <li>-GPRS based Monitoring System.</li>
 <li>-Sening product weights to server.</li>
 - <a style="color:navyblue;font-size:15px;" 
href ='#pi_project_link'> Details </a></li>
</ul>

<h6> Academic Project and Thesis (Undergrad) </h6>
<ul>
 <li>-Remote rescue robot</li>
 <li>-Car velocity logging system</li>
 - <a style="color:navyblue;font-size:15px;" 
href ='#buet_project_link'>Details </a></li>
</ul>
</div>

<h6> Robotics Projects Personal (Undergrad) </h6>
<ul>
 <li>-Hobby CNC machine.
 <li>-Interfacing OV7670 camera sensor with atmega AVR (Feb-May - 2013)
 <li>-Software platform for controlling Robotic ARM(Apr-Sep,2014)
<li><a href ='#embedded_project_link'>Details </a></li>


<style>

    
h1 {color:grey;}
h3 {color:#48C9B0;}
h4 {color:#9FE2BF;}
h5 {color:lightsalmon;}
h6 {color:royalblue;}
a  {color:navyblue;}

#ul_style {
    color:gray ;
    list-style-type:circle;
    } 
    
#div_style {
background-color:#F8F9F9;
}

#ul_st {
    color : #641E16;
    }
    
</style>




<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {
  box-sizing: border-box;
}

.row {
  display: flex;
}

/* Create three equal columns that sits next to each other */
.column {
  flex: 33.33%;
  padding: 5px;
}
</style>
    
<style>
.columns
{   
    -moz-column-width: 21.5em; /* Firefox */
    -webkit-column-width: 21.5em; /* webkit, Safari, Chrome */
    column-width: 21.5em;
}
/*remove standard list and bullet formatting from ul*/
.columns ul
{
    margin: 20px;
    padding: 0;
    list-style-type: none;
}
/* correct webkit/chrome uneven margin on the first column*/
</style>

<div>
<h1  style="color:grey;" >
<a id="ann_link"></a> 
   Neural Network Implementation from scratch</h1></br>
<h3> 1. Overview :  </h3>
</div>

<div id=div_style>
   <p>Here I  have implemented Neural Network of 3 Layers. I have implemented A Layes Class and functions 
   for Forward propagation,backward propagation and updating weights. I just tested it Banknote dataset. It was fitting good.</p>
<ul id=ul_style>
        <b><I>
        <li>Banknote Dataset used for testing</li>
        <li>Forward Propagation</li>
        <li>Backward Propagation</li>
        <ul>
        <li>Calculating Gradients </li>
        <li>Updating weights</li>
        </ul>
        </I></b>
</ul>
</div>

<h3> 2. ANN Flow Chart : (Open Image in new tab for full resolution)</h3>


<img src="docs/Algorihms/NN_Brief.jpg" align="left" alt="Schematics" width="98%" />
<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/ANN_From_Scratch_modular_class-V1.0.ipynb> GitHub : NN Implementation from scratch - Notebook </a>

<iframe src="ANN.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>

<h5> 3. Result of ANN implementation for Bank Note data- a.Precision, Recall and Accuracy and b.True Label vs Prediction </h5>

<img src="docs/Results/banknote_ann.png" align="left"
     title="(Open Image in new tab for good resolution)" width="30%">
<img src="docs/Results/banknote_ann_pred.png" align="left"
     title="(Open Image in new tab for good resolution)" width="70%">


<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>

<h1  style="color:grey;" >
<a id="id3_link"></a> 
   Descision Tree, SVM, Logistic Regression Implementation from scratch</h1>

<h3> 1. Overview </h3>
<div id=div_style>
<ul id=ul_style>
<li><b><I>Dataset     :</li></b></I> Titanic and irish dataset was used for testing ID3.
<li><b><I> Steps      :</li></b></I>
                  <ul>
                  <li>Continuous data spliting based on information gain </li>
                  <li>Information Gain Calculation  </li>
                  <li>ID3 Algorithm according to flowchart </li>
                  </ul>

<li><b><I> Tuning     :</li></b></I> Reduced Error Pruining.
<li><b><I> Prediction :</li></b></I> Accuracy,Precision,Recall Reporting.
</ul>
</div>

<h3> 2.1 ID3 Flow Chart of Implementation (Open Image in new tab for full resolution) </h3>

<img src="docs/Algorihms/ID3_Simple.jpg" align="left"
     title="(Open Image in new tab for full resolution)
" width="100%">
    
<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/ID3_with_continuous_feature_support-V1.0.ipynb>GitHub : ID3 Implementation from scratch - Notebook</a>

<iframe src="ID3.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>

<h5> 3. Result of ID3 implementation for iris data - a.Precision, Recall and Accuracy and b.True Label vs Prediction - </h5>

<img src="docs/Results/iris_ID3.png" align="left"
     title="(Open Image in new tab for good resolution)
" width="30%" >

<img src="docs/Results/irispred.jpg" align="center"
     title="(Open Image in new tab for good resolution)
" width="60%" >

<h3> 2.2 (a)SVM  (b)Logistic Regression Flow Chart of Implementation (Open Image in new tab for full resolution) </h3>
<img src="docs/Algorihms/Logistic Regression.jpg" align="center"
     title="(Open Image in new tab for full resolution)
" width="100%">

<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/SVM.ipynb>(a) GitHub : SVM Implementation from scratch - Notebook </a>

<iframe src="SVM.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>

<h5> 3.1 Result of SVM implementation for banknote data - a.Precision, Recall and Accuracy and b.True Label vs Prediction </h5>

<img src="docs/Results/banknote_svm.png" align="left"
     title="(Open Image in new tab for good resolution)
" width="30%" >

<img src="docs/Results/banknote_svm_pred.png" align="center"
     title="(Open Image in new tab for good resolution)
" width="60%" >

<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/Logistic_Regression.ipynb> (b) GitHub : Logistic Regression Implementation from scratch - Notebook</a>

<iframe src="LogReg.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>

<h5> 3.2 Result of Logitic Regression implementation for banknote data - a.Precision, Recall and Accuracy and b.True Label vs Prediction </h5>

<img src="docs/Results/banknote_logr.png" align="left"
     title="(Open Image in new tab for good resolution)
" width="30%" >

<img src="docs/Results/banknote_logr_pred.png" align="center"
     title="(Open Image in new tab for good resolution)
" width="60%" >


<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>

<div>
<br><h1  style="color:grey;" >
<a id="naive_bayes_link"></a> 
   Naive Bayes and KNN Implementation for text classification</h1></br>
</div>

<h3> 1. Overview </h3>
<div id=div_style>
<ul id=ul_style>
<li><b><I> Dataset     :</li></b></I> Archived data from stack exchange is used for classification.
<li><b><I> Steps      :</li></b></I>
                  <ul>
                  <li>Text Preprocessing was done with raw python without nltk. </li>
                  <li>Calculating Attribute(word) Probabilities given each class  </li>
                  <li>Calculating Attribute(word) Probabilities given each Samples  </li>
                  <li>Applying Bayes Theorem for getting class probalities </li>
                  <li>Max class probality is the predicted label </li>
                  </ul>
<li><b><I> Steps      :</li></b></I>
                  <ul>
                  <li>Text Preprocessing was done with raw python without nltk. </li>
                  <li>Calculating Attribute Probability vector for each sample  </li>
                  <li>Calculating Cosine/Eucleidian/Humming Distances from test sample to each of train sample </li>
                  <li>Taking labels of K min distance samples from train data  </li>
                  <li>Taking mode of K labels as prediction (tie breaks by random choise) </li>
                  </ul>
<li><b><I> Tuning     :</li></b></I> The model was Tuned for best alpha values.
<li><b><I> Prediction :</li></b></I> Accuracy,Precision,Recall Reporting.
</ul>
</div>

<h3> 2. (a)Naive Bayes (b)K Nearest Neighbour algorithm Flow Chart</h3>

<img src="docs/Algorihms/Naive_Bayes.jpg" align="left"
     title="Open Image in new tab for good resolution" width="50%" >
<img src="docs/Algorihms/KNN_Algorithm.jpg" align="left"
     title="Open Image in new tab for good resolution" width="50%" >

<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/Naive_Bayes_Stack_Exchange.ipynb> (a) GitHub : Naive Bayes Implementation from scratch - Notebook </a>
 
<iframe src="NB.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>

<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Machine_Learning_Algo_From_Scratch/KNN_Stack_Exchange.ipynb> (b) GitHub : KNN Implementation from scratch - Notebook</a>

<iframe src="KNN.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>

<h5> 3.1 Result of Naive Bayes implementation for Stack Exchange data - a.Precision Recall and Accuracy b.True Label vs Prediction <h5>
    
<img src="docs/Results/Stack_Exchange_NB_.png" align="left"
     title="(Open Image in new tab for full resolution)" width="30%" >

<img src="docs/Results/Stack_Exchange_NB_pred.jpg" align="left"
     title="(Open Image in new tab for full resolution)" width="70%" >

<h5> 3.2 Result of KNN implementation for Stack Exchange data- a.Precision Recall and Accuracy b.True Label vs  Prediction <h5>

<img src="docs/Results/Stack_Exchange_KNN.png" align="left"
     title="(Open Image in new tab for full resolution)" width="30%" >

<img src="docs/Results/Stack_Exchange_KNN_pred.jpg" align="left"
     title="(Open Image in new tab for full resolution)" width="70%" >


<a style="color:navyblue;font-size:15px;" href ='#table_of_content_link'>Go Back to Table of Content</a>


<div>
<br><h1  style="color:grey;" >
<a id="house_price_link"></a> 
    House Price Prediction :: Data Pre-Processing and Hiper-Parameter Tuning</h1></br>
<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/ANN_Tensorflow_Kaggle_Houseprice_prediction.ipynb>  Notebook : House Price Prediction - Notebook (Project Presentation and Code Link) </a>
<iframe src="house_price.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>
</div>

<h3> 1. Overview :</h3>
     
<div id=div_style>
    <ul id=ul_style>
        <b><I>
        <li>House Price Dataset from kaggle.com</li>
        <li>Data Preprocessing</li>
        <li>ANN Class with tensorflow low level API </li>
        <li>Hiperparameter Tuning</li>
        <li>All the graphs of Data preprocessing and Hiperparameter Tuning can be found in <a href = https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/ANN_Tensorflow__Kaggle_Houseprice_prediction.ipynb> Notebook </a></li>
        </I></b>
    </ul>
<details false>
    <summary><b><I><font color="#3498DB"> Click to expand</font></I></b></summary>
    
<ul style="list-style-type:none;" >
<li><b> Dataset : </b> House Price Dataset of kaggle.com </li>
<li><b> Data Preprocessing : </b></li>
            <ul>
            <li> 2.0 Correlation Analysis </li>
            <li> 2.1 Outlier handling </li>
            <li> 2.2 Missing value Handling </li>
            <li> 2.3 Catagorical to numerical conversion </li>
            <li> 2.4 Unskewing while needed </li>
            <li> 2.5 Data scaling </li>
            </ul>
<li><b> ANN Class with tensorflow low level API : </b></li>
            <ul>
            <li> Method train </li>
            <li> Method predict </li>
            <li> Method save weights </li>
            <li> Method load weights </li>
            </ul>
<li> Each preprocessing step's effectiveness is checked by simple linear regression </li>
<li><b> Hiperparameter Tuning : </b></li>
            <ul>
            <li> 4.1 Layer Optimization </li>
            <li> 4.2 Optimizer grid search (Each optimizer for learning rates and epochs) </li>
            <li> 4.3 Learning Rate Optimization </li>
            <li> 4.4 Epochs, Early stopping, Regularization constan optimization </li>
            <li> 4.5 Activation optimazation </li>
            <li> 4.6 Batch size optimization </li>
            </ul>
</ul>
<ul>
<li> Cross-Validation with 3 fold was done for overfitting testing </li>
<li> Test result was submitted in kaggle.com for evaluation screenshot can be found at result section </li>
<li> All the graphs of Data preprocessing and Hiperparameter Tuning can be found in <a href = https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/ANN_Tensorflow__Kaggle_Houseprice_prediction.ipynb> Notebook </a>  </li>
</ul>    
</details>
</div>

<h3> 2. Project Flow Chart :</h3>
<img src="docs/Algorihms/kaggle_hp.jpg" align="center"
     title="Open Image in new tab for good resolution" width="100%" height="1000">
     
<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>


<div>
<br><h1  style="color:navyblue;" >
<a id="sakura_link"></a> 
    Japanese Job Entrance Problem :: Shakura Bloom Prediction</h1></br>
<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/Sakura_TF_NN_Report.ipynb>   Notebook : Shakura Bloom Prediction - Notebook (Project Presentation and Code Link) </a>
<iframe src="sakura.html" name="targetframe" allowTransparency="true" scrolling="yes" frameborder="0" height="300" width="100%" ></iframe>

</div>


<h3> 1. Overview </h3>
<div id=div_style>
<ul id=ul_style>
        <b><I>
        <li>Weather data from japanese meteorological agencym</li>
        <li>Feature Extraction and Data Preprocessing</li>
        <li>ANN Class with tensorflow low level API </li>
        <li>Hiperparameter Tuning</li>
        <li>All the graphs of Data preprocessing and Hiperparameter Tuning can be found in <a href = https://github.com/irfanhasib0/Machine-Learning/blob/master/Kaggle/Sakura_TF_NN_Report.ipynb> Notebook </a></li>
        </I></b>
    </ul>
<details false>
        <summary><b><I><font color="#3498DB"> Click to expand</font></I></b></summary>
<ul style="list-style-type:none;">    
<li><b> Dataset : </b></li> Weather data from japanese meteorological agency
<li><b> Steps :</b></li>
                 <ul>
                 <li> Feature Analysis and Data Preprocessing </li>
                 <li> Implementing Nural Network with tensorflow low level API. </li>
                 <li> Hiper-Parameter Tuning for ANN </li>
                 </ul>
<li><b> Data Preprocessing : </b></li>
         <ul>
         <li><b> Converting features of a whole year(365/366 samples) to a single vector for adding feature vector.  </b></li>
                    <ul>
                     <li>  Converting all the 365 days daya to a single feature</li>
                        <ul>
                        <li> 1. Days after bloom can be ignored. As weather before bloom can effect blooom date </li>
                        <li> 2. Days days before Dj- 'Last day of hibernation' is less significant </li>
                        <li> 3. Feature with high correlation i.e max temp, hr1 preci can be specially considered for processing </li>
                        </ul>
                        <li>  Converting all the 365 days daya to a single feature </li>
                         <ul>
                         <li> 1. Mean of first 90 days </li>
                         <li> 2. Mean of 30-90 th days </li>
                         <li> 3. mean of Dj-Dj+45 days </li>
                         <li> 4. Mean Dj-Dj+60 days </li>
                         <li> 5. Mean of Dj-Dj+75 </li>
                         </ul>
                     </ul>
        <li><b> Feature Selection </b></li>
               <ul>
               <li> Co-relation analysis </li>
               <li> Accuracy of the linear regression after adding each feature </li>
               </ul>
       <li><b> Highly Corelated feaure specially anayzing and processing.  </b></li>
       <li><b> Outlier analysis.  </b></li>
       <li><b> Scaling the data considering interquantile range  </b></li>
       </ul>
<li><b> Hiperparameter Tuning : </b></li>
             <ul>
             <li> 1. Layer Optimization </li>
             <li> 2. Optimizer grid search (Each optimizer for learning rates and epochs) </li>
             <li> 3. Learning Rate Optimization </li>
             <li> 4. Epochs, Early stopping, Regularization constant optimization. </li>
             </ul>
</ul>    
</details>


<h3> 2. Project Flow Chart :</h3>
<img src="docs/Algorihms/sakura_jp.jpg" align="center"
     title="Open Image in new tab for good resolution" width="70%">
<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>
</div>

<div>
<br><h1  style="color:grey;" >
<a id="dqn_ddpg_link"></a> DQN , DDPG, PPO A2C Implementation from scratch</h1></br>
<h5> GitHub Code Links :
<ul>
<li><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/RL-Algorithms/blob/master/1.0_Multi_Arm_Bandit.ipynb>  Multi Arm Bandit - Notebook </a></li>
<li><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/RL-Algorithms/blob/master/2.1_Value_Iteration.ipynb>   Value Iteration - Notebook  </a></li>
<li><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/RL-Algorithms/blob/master/3.0_DQN_Mountain_Car.ipynb>  Mountain Car with DQN - Notebook  </a></li>
<li><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/RL-Algorithms/blob/master/4.0_DDPG_Pendulum-mod.ipynb> Pendulum with DDPG - Notebook</a></li>
<li><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/RL-Algorithms/blob/master/5.1_PPO_Bipedal_Walker-mod.ipynb.ipynb>  Bipedal Walker with PPO - Notebook</a></li>
<li><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/RL-Algorithms/blob/master/5.1_PPO_Lunar_Lander.ipynb>  Lunar Lander with PPO - Notebook </a></li>
</ul>
</div>


<h3> 1.  Overview </h3>

<div id=div_style> 
<ul id=ul_style>
    <li><b><I> DQN Environments </li></b></I>  OpenAI gym --> Mountain Car ENvironment
    <li><b><I> DQN Environments </li></b></I>  OpenAI gym --> Pendulumn Environment
    <li><b><I> DQN Steps </li></b></I>
    <ul>
    For each time step of a episode 
    <li> action : explore -> random | or exploit -> max Q value based on epsilon decay </li>
    <li> play one step -> store experience </li>
    <li> sample minibatch </li>
    <li> target Q values : reward+gamma*Q_network(new_state,new_actions) </li> 
    <li> train Q network </li>
    </ul>
    <li><b><I> DDPG Steps </li></b></I>
    <ul>
    For each time step of a episode 
    <li>actor(state): action + OU Noise </li>
    <li>play one step -> store experience </li>
    <li>sample minibatch </li>
    <li>target Q values : reward+gamma*target_critic(new_state,new_actions) </li>
    <li>actor loss from critic(state,action)</li>
    <li>train actor and critic </li>
    <li>target networks : networks*tau + (1-tau)*target_networks</li>
    </ul>
</ul>
</div>

<div>
<h3> 2. Detailed Flow Chart for DQN and DDPG : (Please open in New tab for proper resolution) </h3>

<img src="docs/Algorihms/DQN_Brief.jpg" align="left"
     title="Open Image in new tab for good resolution" width="50%" >
<img src="docs/Algorihms/DDPG_Brief.jpg" align="left"
     title="Open Image in new tab for good resolution" width="50%" ></br>
<img src="docs/Algorihms/PPO_A2C_Brief.jpg" align="left"
     title="Open Image in new tab for good resolution" width="100%" >
</div>

<div>
<h3> 3. Results </h3>

* a. Results DQN on Mountain Car
* b. Results DDPG on Pendulum 
* c. Results PPO on Bipedal Walker
* d. Results PPO on Lunar Lander

<img src="docs/Results/dqn_mcar.png" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
<img src="docs/Results/ddpg_pendulum.png" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
<img src="docs/Results/bipedal_ppo.png" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
<img src="docs/Results/lunar_ppo.png" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
     
<img src="docs/Results/mc.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
<img src="docs/Results/pend.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
<img src="docs/Results/bipedal_ppo.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
<img src="docs/Results/lunar_ppo.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="25%" />
     
<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>
</div>

<div>
<br><h1  style="color:grey;" >
<a id="ilqr_mpc_link"></a> 
   ILQR and MPC :: Implementation from scratch for self driving car simulator</h1></br>

<br><a style="color:navyblue;font-size:15px;" href=https://github.com/irfanhasib0/Control-algorithms/blob/master/MPC_GYM_CAR_RACING_V0/ilqr_on_airsim_env_map_tracker_rel-exp.ipynb> Notebook : ILQR Implementation - Notebook (Project Presentation and Code Link) </a></br>

<a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/Control-algorithms/blob/master/MPC_GYM_CAR_RACING_V0/mpc_on_car_env_map_tracker_rel.ipynb> Notebook : MPC Implementation - Notebook (Project Presentation and Code Link)</a>
</div>

<div id=div_style>
<h3> 1. Overview : </h3>
<ul id=ul_style>
        <li><b><I> Simulation Platform : </b></I></li>
            <ul style="list-style-type:circle;">
                <li> AIRSIM by Microsoft Inc. </li>
                <li> OpenAI GYM - Car Environment was tested</li>
            </ul>
        <li><b><I>IO :</b></I></li> Input --> Map Points , Output --> Steering Angle, Acclelation, Brake
        <li><b><I>MAP to Trajectory : Steps : </b></I></li>
            <ul style="list-style-type:circle;">
                <li>  Environment Module </li>
                <li>  Map Tracker Module </li>
                <li>  Data Preprcessor Module</li>
            </ul>
        <li><b><I> Optimaization Algorithm : Trajectory to Optimal Steering,Accleration,Brake : </b></I></li>
            <ul>
                <li>iLQR using Raw Python and Numpy</li> 
                <li>MPC using Python , Numpy and CVXPY</li>
            </ul>
    </ul>

      
<details false>
    <summary><b><I><font color="#3498DB"> Click to expand</font></I></b></summary>
<ul style="list-style-type:none;">
 
<li><b>  Simulation Environment : </b></li>
    <ul>
    <li> AIRSIM a Car Simulator by Microsoft </li>
    <li> OpenAI gym Car ENnvironment. </li>
    </ul>
<li><b> Original Paper of ILQR Part of the Project </b></li>
                    <ul style="list-style-type:none;"><li>
                    Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization By - Y Tassa</li></ul>
<li><b>  I0 :   </b></li>
                    <ul style="list-style-type:none;"><li>
                        Input --> Map Points , Output --> Steering Angle, Acclelation, Brake</li></ul>
<li><b>  Steps :  </b></li>
                 <ul>
                 <li> Map Tracker Module </li>
                         <ul>
                         <li> Input : Takes Map points as Input
                         <li> Trajectory Queue : Gets N next points from map points to follow according to car position and orientation.
                         <li> Update Trajectory Queue : Update the queue of points to follow with moving car position and orientation.
                         <li> Output : Next N points to follow from Trajectory Queue
                         </ul>
                <li> Data Preprcessor Module : </li>
                         <ul>
                         <li> Input : Trajectory points from Map tracker module.
                         <li> Full State Calculation : Calculates yaw from ref points and target velocity.
                         <li> Relative Trajectory calculation : Relative Co-ordinates calculation as Car position nad yaw as                         origin.
                         <li> Output : Adjusts no of points to track in the refference trajectory accrding to current velocity.  
                         </ul>
                 <li> ILQR Module </li>
                          <ul>
                           <li> Input : Refferance trajectory to follow from Data Processor module. </li>
                          <li> Output : Calculate Optimal steering angle and accelaration with ILQR algorithm shown below. </li>
                          </ul>
                 OR,
                 <li> MPC Module</li>
                              <ul>
                              <li> Input : Refferance trajectory to follow from Data Processor module. </li>
                              <li> Output : Calculate Optimal steering angle and accelaration with MPC algorithm shown below. </li>
                              </ul>
                 </ul>
</ul>    
</details>
</div>

<div>
<h3 style='color:teal'> MPC modeling </h3>
$\begin{align*}
State\;Space :z &= [x, y, v,\phi]        &where,x: x-position, y:y-position, v:velocity, φ: yaw angle\\
Action\;Space:u &= [a, \delta]          &where,a: accellation, δ: steering angle\\
\end{align*}$

<h3 style='color:teal'>Cost and Constraints : </h3>
$\begin{align*}
&\color{navy} {Cost :}\\
&min\ Q_f(z_{T,ref}-z_{T})^2+Q\Sigma({z_{t,ref}-z_{t}})^2+R\Sigma{u_t}^2+R_d\Sigma({u_{t+1}-u_{t}})^2\\
&z_{ref}\;:\;target\;states\\
\end{align*}$
$\begin{align*}
\color{navy} {Constraints :}&\\
z_{t+1}&=Az_t+Bu+C\\
Maximum\;steering\; speed &=abs[u_{t+1}-u_{t}]{<}du_{max}\\
Maximum\;steering\;angle &=u_{t}{<}u_{max}\\
Initial\;state &= z_0 = z_{0,ob}\\
Maximum\;and\;minimum \;speed &= v_{min} {<} v_t {<} v_{max}\\
Maximum\;and minimum\;input &=u_{min} {<} u_t {<} u_{max}\\
\end{align*}$

<h3 style='color:teal'>State Space model for Car system</h3>

$\begin{align*}
z_{t+1}=Az_t+Bu+C\;where,\\
\end{align*}$
$\begin{equation*}
A = 
\begin{bmatrix} 
1 & 0 & cos(\bar{\phi})dt & -\bar{v}sin(\bar{\phi})dt\\
0 & 1 & sin(\bar{\phi})dt & \bar{v}cos(\bar{\phi})dt \\
0 & 0 & 1 & 0 \\
0 & 0 &\frac{tan(\bar{\delta})}{L}dt & 1 \\
\end{bmatrix}
\end{equation*}$
$\begin{equation*}
B =
\begin{bmatrix} 
0 & 0 \\
0 & 0 \\
dt & 0 \\
0 & \frac{\bar{v}}{Lcos^2(\bar{\delta})}dt \\
\end{bmatrix}
C =
\begin{bmatrix} 
\bar{v}sin(\bar{\phi})\bar{\phi}dt\\
-\bar{v}cos(\bar{\phi})\bar{\phi}dt\\
0\\
-\frac{\bar{v}\bar{\delta}}{Lcos^2(\bar{\delta})}dt\\
\end{bmatrix}
\end{equation*}$

<details> 
    <h3 style='color:teal' >Car Steering Model with Linearization </h3>
    <summary style='color:blue'><h4><b><I> Expand to see derivation</b></I></h4></summary>
$\begin{align*}
 \dot{x} &= vcos(\phi)                      &\\
 \dot{y} &= vsin(\phi)       & \dot{z} &=\frac{\partial }{\partial t} z = f(z, u) = A'z+B'u\\
 \dot{v} &= a                               &\\
 \dot{\phi} &= \frac{vtan(\delta)}{L}       &\\
\end{align*}$
    
where

$\begin{equation*}
A' =
\begin{bmatrix}
\frac{\partial }{\partial x}vcos(\phi) & 
\frac{\partial }{\partial y}vcos(\phi) & 
\frac{\partial }{\partial v}vcos(\phi) &
\frac{\partial }{\partial \phi}vcos(\phi)\\
\frac{\partial }{\partial x}vsin(\phi) & 
\frac{\partial }{\partial y}vsin(\phi) & 
\frac{\partial }{\partial v}vsin(\phi) &
\frac{\partial }{\partial \phi}vsin(\phi)\\
\frac{\partial }{\partial x}a& 
\frac{\partial }{\partial y}a& 
\frac{\partial }{\partial v}a&
\frac{\partial }{\partial \phi}a\\
\frac{\partial }{\partial x}\frac{vtan(\delta)}{L}& 
\frac{\partial }{\partial y}\frac{vtan(\delta)}{L}& 
\frac{\partial }{\partial v}\frac{vtan(\delta)}{L}&
\frac{\partial }{\partial \phi}\frac{vtan(\delta)}{L}\\
\end{bmatrix}
　=
\begin{bmatrix}
0 & 0 & cos(\bar{\phi}) & -\bar{v}sin(\bar{\phi})\\
0 & 0 & sin(\bar{\phi}) & \bar{v}cos(\bar{\phi}) \\
0 & 0 & 0 & 0 \\
0 & 0 &\frac{tan(\bar{\delta})}{L} & 0 \\
\end{bmatrix}
\end{equation*}$

$\begin{equation*}
B' =
\begin{bmatrix}
\frac{\partial }{\partial a}vcos(\phi) &
\frac{\partial }{\partial \delta}vcos(\phi)\\
\frac{\partial }{\partial a}vsin(\phi) &
\frac{\partial }{\partial \delta}vsin(\phi)\\
\frac{\partial }{\partial a}a &
\frac{\partial }{\partial \delta}a\\
\frac{\partial }{\partial a}\frac{vtan(\delta)}{L} &
\frac{\partial }{\partial \delta}\frac{vtan(\delta)}{L}\\
\end{bmatrix}
　=
\begin{bmatrix}
0 & 0 \\
0 & 0 \\
1 & 0 \\
0 & \frac{\bar{v}}{Lcos^2(\bar{\delta})} \\
\end{bmatrix}
\end{equation*}$
<p>
<br>Forward Euler Discretization with sampling time dt.
Using first degree Tayer expantion around zbar and ubar</br>
</p>
$\begin{align*}
z_{k+1}&=z_k+f(z_k,u_k)dt\\
z_{k+1}&=z_k+(f(\bar{z},\bar{u})+A'z_k+B'u_k-A'\bar{z}-B'\bar{u})d\\
z_{k+1}&=(I + dtA')z_k+(dtB')u_k + (f(\bar{z},\bar{u})-A'\bar{z}-B'\bar{u})dt\\
z_{k+1}&=Az_k+Bu_k +C\\
\end{align*}$
<p><br>
So,
</br></p>
$\begin{equation*}
A = (I + dtA')
=
\begin{bmatrix} 
1 & 0 & cos(\bar{\phi})dt & -\bar{v}sin(\bar{\phi})dt\\
0 & 1 & sin(\bar{\phi})dt & \bar{v}cos(\bar{\phi})dt \\
0 & 0 & 1 & 0 \\
0 & 0 &\frac{tan(\bar{\delta})}{L}dt & 1 \\
\end{bmatrix}
\end{equation*}$
$\begin{equation*}
B = dtB'
=
\begin{bmatrix} 
0 & 0 \\
0 & 0 \\
dt & 0 \\
0 & \frac{\bar{v}}{Lcos^2(\bar{\delta})}dt \\
\end{bmatrix}
\end{equation*}$

$\begin{equation*}
C = (f(\bar{z},\bar{u})-A'\bar{z}-B'\bar{u})dt\\
= dt(
\begin{bmatrix} 
\bar{v}cos(\bar{\phi})\\
\bar{v}sin(\bar{\phi}) \\
\bar{a}\\
\frac{\bar{v}tan(\bar{\delta})}{L}\\
\end{bmatrix}
-
\begin{bmatrix} 
\bar{v}cos(\bar{\phi})-\bar{v}sin(\bar{\phi})\bar{\phi}\\
\bar{v}sin(\bar{\phi})+\bar{v}cos(\bar{\phi})\bar{\phi}\\
0\\
\frac{\bar{v}tan(\bar{\delta})}{L}\\
\end{bmatrix}
-
\begin{bmatrix} 
0\\
0 \\
\bar{a}\\
\frac{\bar{v}\bar{\delta}}{Lcos^2(\bar{\delta})}\\
\end{bmatrix}
)\\
=
\begin{bmatrix} 
\bar{v}sin(\bar{\phi})\bar{\phi}dt\\
-\bar{v}cos(\bar{\phi})\bar{\phi}dt\\
0\\
-\frac{\bar{v}\bar{\delta}}{Lcos^2(\bar{\delta})}dt\\
\end{bmatrix}
\end{equation*}$
</details>

<h3 style='color:teal'>Iterative Linear Quadratic Regulator :</h3>
<h4 style='color:navy'> 1. U as a function of Previous U , X and Previous X : </h4>
$\begin{equation}
u{'}(i) = u(i) + k(i) +K(i)(x{'}(i) - x(i))\\
\end{equation}$

<details>
<summary><h4 style='color:navy'> 2. Calcualting K and k : (Expand to see) </h4></summary>
$\begin{align*}
\color{navy}{Step\;A.}\;&\color{navy}{Getting\;Q[i]s\;from\;l[i]s\;and\;V[i]s:}\\\\
Q_x &= l_x(t) + f_x(t).T, V_x \\
Q_u &= l_u(t) + f_u(t).T, V_x \\
Q_{xx} &= l_{xx}(t) + f_x(t).T * (V_{xx} * f_x(t) )  \\
Q_{ux} &= l_{ux}(t) + f_u(t).T * (V_{xx} * f_x(t) )) \\
Q_{uu} &= l_{uu}(t) + (f_u(t).T *(V_{xx} * f_u(t) ) \\\\
\end{align*}$
$\begin{align*}
&\color{navy}{Step\;B.\;Getting\;K[i]s\;from\;Q[i]s:}\\\\
&Q_{uu\_evals}, Q_{uu\_evecs} = np.linalg.eig(Q_{uu})\\
&Q_{uu\_evals}[Q_{uu_evals} < 0] = 0.0\\
&Q_{uu\_inv} = Q_{uu_evecs} * \\&(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T)\\              
&k(t) = -1. * (Q_{uu\_inv} * Q_u)\\
&K(t) = -1. * (Q_{uu\_inv} * Q_{ux})
\end{align*}$
$\begin{align*}
&\color{navy}{Step\;C.\;Getting\;V[i+1]s\;from\;K[i]s:}\\\\
V_x &= Q_x - (K[t].T * (Q_{uu} * k[t]))\\
V_{xx} &= Q_{xx} - (K[t].T * (Q_{uu} * K[t]))\\\\\\\\\\
\end{align*}$
<h4 style='color:navy'> 3. Step A , B and C run iteratively fo find optimal K and k.</h4>
</details>
</div>

<div>
<h4 style='color:navy'> 4. For detail derivation please see the paper :</h4> <p style='color:gray'>Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization
by -Yuval Tassa 
    <br> citations(369) according to February 27 2020, Published on - 2012</p>
    
<h3>  2. Project Flow Chart : </h3>
<img src="docs/Algorihms/MPC_Brief.jpg" align="center"
     title="Open Image in new tab for good resolution" width="100%" >

<details>
     <summary><b><I><font color="#3498DB"> Click to expand</font></I></b></summary>
<img src="docs/Algorihms/MPC_Detail.jpg" align="center"
     title="Open Image in new tab for good resolution" width="80%" >
</details>

     
<h3> 3. Results (ILQR) : </h3>
* OpenAI Gym Car Environment 
* Airsim City Space Environment 
* Airsim Neighbourhood Environment 

<img src="docs/Results/rec_car_env.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="30%" />
<img src="docs/Results/fig_car_env.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="30%" />
<img src="docs/Results/airsim_cs.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="30%" />
<img src="docs/Results/fig_cs.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="30%" />
<img src="docs/Results/airsim_nh.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="30%" />
<img src="docs/Results/fig_nh.gif" align="center"
     title="(Open Image in new tab for full resolution)" width="30%" />
     
     
<h3>Inspired from - </h3>

-[AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics/blob/f51a73f47cb922a12659f8ce2d544c347a2a8156/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py#L247-L301)

<h3> Reference </h3>

- [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics/blob/eb6d1cbe6fc90c7be9210bf153b3a04f177cc138/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py#L80-L102)
- Synthesis and Stabilization of Complex Behaviors through
Online Trajectory Optimization
by -Yuval Tassa

<h3> Appendix : Map Tracker and iLQR</h3>
<img src="docs/Algorihms/map_tracker.jpg" 
     title="Open Image in new tab for good resolution" align="left" width="40%" >
<img src="docs/Algorihms/iLQR_Algorithm_up.jpg" 
     title="Open Image in new tab for good resolution" aligh="left" width="60%" >
<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>
</div>

<div>
<br><h1  style="color:grey;" >
<a id="yolo_link"></a> 
   Yolo-V2.0 with KERAS and Tensorflow for Object detection</h1></br>
<br><a style="color:navyblue;font-size:15px;" https://github.com/irfanhasib0/CNN-Projects/blob/master/CNN_Basic/Minimal_yolo_v_3_functioned-exp_mod-v-1.0.ipynb> GitHub : Yolo-V3 Implementation Notebook </a></br>
</div>

<img src="docs/Algorihms/yolo_algo.jpg" align="center"
     title="(Open Image in new tab for full resolution)" width="80%" />
     <img src="docs/Results/yolo_out_1.gif" align="left"
     title="(Open Image in new tab for full resolution)" width="48%" />
     <img src="docs/Results/yolo_out_2.gif" align="right"
     title="(Open Image in new tab for full resolution)" width="48%" />
   
<div>
    
<br><h1  style="color:grey;" >
<a id="unet_link"></a> 
   U-Net with KERAS for City Space Dataset</h1></br>
<br><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/CNN-Projects/blob/master/as_unet_seg-cs.ipynb> GitHub :Unet for segmenting City Space Dataset </a></br>
</div>

<img src="docs/Results/unet_pred.jpg" align="center"
     title="(Open Image in new tab for full resolution)" width="100%" />
<img src="docs/Results/unet_res.jpg" align="center"
     title="(Open Image in new tab for full resolution)" width="50%" />
 <a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>    

<div>
<h1  style="color:grey;" >
<a id="flownet_link"></a> 
  Flow-Net with KERAS and Tensorflow</h1></br>
<br><a style="color:navyblue;font-size:15px;" href= https://github.com/irfanhasib0/CNN-Projects/blob/master/CNN_Basic/visual_odometry_deepvo%20-%20M-Flow%20-%20V-1.0.ipynb> Notebook :FlowNet for Optical Flow Kitti Dataset </a></br>
</div>
Top row : Train ; Bottom Row : Validation ;
Left : Ground Truth ; Right : Prediction
<img src="docs/Results/of_true-1.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
<img src="docs/Results/of_pred-1.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
     
<img src="docs/Results/of_true-2.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
<img src="docs/Results/of_pred-2.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
 <a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a> 

<h1  style="color:grey;" >
<a id="disp_link"></a> 
  Monocular Disparity with CNN based Encode-Decoder with KERAS and Tensorflow</h1></br>
<a style="color:navyblue;font-size:15px;" https://github.com/irfanhasib0/CNN-Projects/blob/master/CNN_Basic/visual_odometry_deepvo%20-%20M-Depth%20-%20V-2.0.ipynb> Notebook : Single View Depth Estimation </a>
</div>

Top row : Train ; Bottom Row : Validation ;
Left : Ground Truth ; Right : Prediction
<img src="docs/Results/disp_train_gt.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
<img src="docs/Results/disp_train_pred.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
     
<img src="docs/Results/disp_val_gt.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
<img src="docs/Results/disp_val_pred.png" align="left"
     title="(Open Image in new tab for full resolution)" width="50%" />
 <a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a> 

<div>
<br><h1  style="color:grey;" >
<a id="ros_rrbot_link"></a> 
   ROS : Simple two linked robot inspired from rrbot</h1></br>
<br>
</div>

<ul>
<li> <a href= "https://github.com/irfanhasib0/ros_ws/tree/master/src/rrbot/rrbot_description">URDF Link</a> </li>
<li> <a href="https://github.com/irfanhasib0/ros_ws/tree/master/src/rrbot/rrbot_control">Controller Link </a> </li>
<li> <a href="https://github.com/irfanhasib0/ros_ws/tree/master/src/rrbot/rrbot_gazebo"> Gazebo Link</a> </li>
<li> <a href="https://youtu.be/lJbyy89X7gM">Vedio Link</a> </li>
</ul>

<div>
<br><h1  style="color:grey;" >
<a id="pi_project_link"></a> 
   Embedded System Projects : Pi Labs BD LTD</h1></br>
<br>
</div>


All these projects I did as an employee of Pi Labs BD Ltd. www.pilabsbd.com


<img src="docs/old/vault_sequrity.jpg" align="left" alt="(Open Image in new tab for full resolution)" width="50%" />
     
<img src="docs/old/safe_box.jpg"  alt="(Open Image in new tab for full resolution)" width="50%" />
     
<img src="docs/old/syringe_pump.jpg" align="left" alt="(Open Image in new tab for full resolution)" width="50%" />
     
<img src="docs/old/weight_machine.jpg" alt="(Open Image in new tab for full resolution)" width="50%" />



<div>
<br><h1  style="color:grey;" >
<a id = "buet_project_link"></a> 
   Academic Project and Thesis:</h1></br>
<br>
</div>

* My undergrad project of intrumentation and measurement course
* My undergrad thesis


<img src="docs/old/thesis_project.jpg" alt="(Open Image in new tab for full resolution)" width="80%" />


<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>

<div>
<br><h1  style="color:grey;" >
<a id = "urc_2016_link"></a> 
   University Rover Challenge - 2016</h1></br>
<br>
</div>

### Critical Design Rivew    : [Video Link](https://www.youtube.com/watch?v=MlN-VFj14LE)

<img src="docs/old/URC.jpg" alt="(Open Image in new tab for full resolution)" width="80%" />
<a style="color:navyblue;font-size:15px;" 
href ='#table_of_content_link'>Go Back to Table of Content</a>

<div>
<br><h1  style="color:grey;" >
<a id = "embedded_project_link"></a> 
   Embedded System & Robotics Projects Personal (Undergrad) :</h1></br>
<br>
</div>
<ul>
 <li> Hobby CNC machine ‘Ourtech v 2.0’ and 'Ourtech v1.0', Desktop CNC Machine.<a href = 'https://www.youtube.com/watch?v=xU7YMPpZMYs'>Video Link</a></li>
 <li> Interfacing ov7670 camera sensor with atmega 32 and using object tracking algorithm on AVR platform (Feb-May - 2013)<a href='https://youtu.be/tKWbYbAJSEs'>Video Link</a></li>
 <li> Software platform for controlling Robotic arm with openCV, python and Raspberry pi (Apr-Sep,2014)<a href='https://www.youtube.com/watch?v=hj1Wc6-8-7w'>Video Link</a></li>
</ul>
    
  <img src="docs/old/CNC.jpg" align="center" alt="(Open Image in new tab for full resolution)" width="50%" />
 
  <img src="docs/old/Robot_Arm.jpg" align="left" alt="(Open Image in new tab for full resolution)" width="50%" />
 
  <img src="docs/old/Robot_Arm_Soft.jpg" align="left" alt="(Open Image in new tab for full resolution)" width="50%" />


```python

```
