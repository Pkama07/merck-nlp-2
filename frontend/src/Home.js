// Home.js
import React from 'react';
import './App.css';
import "bootstrap/dist/css/bootstrap.min.css";
import SearchBar from './searchBar';
import Upload from './upload';
import Logo from './logo';
import { Link } from 'react-router-dom';


export default function Home({ renderResults }) {
  return (
    <div className='topBar'>
      <div className='titleBar'>
        <div id='logo'>
          <Link to="/">
            <Logo />
          </Link>
        </div>
        <div id='upload' style={{ marginRight: "2vw" }}>
          <Upload />
        </div>
      </div>

      <div id="title-box">
        <div id="title-text">Bioanalytical Procedure Database System</div>
        <div id="sub-text">The Data Mine - Corporate Partners: Merck NLP</div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <div id='search'>
          <SearchBar renderResults={renderResults} />
        </div>
      </div>


    </div>



  );
}
