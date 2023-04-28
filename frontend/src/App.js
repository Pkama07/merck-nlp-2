import './App.css';
import Home from './Home';
import "bootstrap/dist/css/bootstrap.min.css";
import Results from './results';
import { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function App() {
  const [resultsData, setResultsData] = useState(null);

  const renderResults = (data) => {
    setResultsData(data);
  };

  return (
    <Router>

      <Routes>
        <Route path="/" element={<Home renderResults={renderResults} />} />
        <Route path="/results" element={<Results resultsData={resultsData} renderResults={renderResults} />} />
      </Routes>

    </Router>
  );
}

export default App;
