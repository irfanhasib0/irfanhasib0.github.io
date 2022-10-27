import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <Row>
      <h6> Machine Learning Algorithms from scratch</h6>
      <ul>
          <li><a style="color:navyblue;font-size:15px;" 
                 href ='#ann_link'> Neural Network </a></li>
          <li><a style="color:navyblue;font-size:15px;" 
                 href ='#id3_link'> Decision Tree Algorithms</a></li>
          <li><a style="color:navyblue;font-size:15px;" 
                 href ='#svm_link'> SVM, Logistic Regression</a></li>
          <li><a style="color:navyblue;font-size:15px;" 
                 href ='#naive_bayes_link'> Naive Bayes, KNN</a></li>
       </ul>
       </Row> 

      <iframe src='readme.html' width="100%" height="10000 px"></iframe>
    </div>
  );
}

export default App;
