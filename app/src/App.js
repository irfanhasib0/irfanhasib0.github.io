import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css';
import {useRef, useEffect} from 'react'
import logo from './logo.svg';
import './App.css';
import bg1 from './img/bg1.jpg'

//<a href="readme.html#ann_link"> ANN </a>
function App() {
   const fref = useRef('null')
   useEffect(()=>{
   const iframe = fref.current
   
   iframe.onload = function()
        {
          iframe.style.height = 
          iframe.contentWindow.document.body.scrollHeight + 'px';
          iframe.style.width  = 
          iframe.contentWindow.document.body.scrollWidth+'px';
              
        }
        
        
   //var iframe = document.getElementById('mainframe');
   //var doc = (iframe.contentDocument)? iframe.contentDocument: iframe.contentWindow.document;

   //var anchors = doc.getElementsByTagName('a');
   //for (var i = 0; i < anchors.length; i++){
   //   console.log(anchors[i])
   //   anchors[i].target = '_PARENT';
   //   }
  })
  return (
    <>
      <Row>
      <Col xs={2} style={{ backgroundImage : `url(${bg1})`}}>
      <div > Side bar </div>
      </Col>
      <Col xs={9}>     
      <iframe ref={fref} src={"readme.html"} scrolling="no" width={"100%"} height={"10000 px"}></iframe>
      </Col>
      </Row> 
    </>
  );
}

export default App;
