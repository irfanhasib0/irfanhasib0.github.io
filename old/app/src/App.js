import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css';
import {useRef, useEffect} from 'react'
import logo from './logo.svg';
import './App.css';
import bg1 from './img/bg1.jpg'
import {BsJournalRichtext, BsBook, BsStack, BsVectorPen,BsGem,BsLinkedin,BsFillTerminalFill} from 'react-icons/bs';
import {AiOutlineMail, AiOutlineLogin, AiOutlineLogout, AiOutlineCaretUp, AiOutlineCopy} from 'react-icons/ai';

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
      <div > </div>
      </Col>
      <Col xs={9}>
      <div style={{marginLeft : "80px", marginTop : "10px"}}>
      <BsBook /><a href='https://irfanhasib0.github.io/blogs' target='_blank'>Blog</a>
      &nbsp;&nbsp;<BsBook /><a href='/research.html' target='_blank'>Research</a>
      </div>
      <iframe ref={fref} src={"readme.html"} scrolling="no" width={"1200 px"} height={"10000 px"}></iframe>
      </Col>
      </Row> 
      </>
  );
}

export default App;
