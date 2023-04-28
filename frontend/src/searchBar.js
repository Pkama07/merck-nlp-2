import React, { useState } from 'react';
import axios from 'axios';
import './searchBar.css';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';
import { useNavigate } from 'react-router-dom';
import Button from 'react-bootstrap/Button';


function SearchBar({ renderResults }) {
  const url = process.env.REACT_APP_API_ENDPOINT;
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();
  const [message, setMessage] = useState('');
  const [confirmation, setConfirmation] = useState('');

  const handleSearchTermChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleSubmit = async (event) => {
    setMessage('');
    setConfirmation('');
    event.preventDefault();
    const trimmedSearchTerm = searchTerm.trim();
    const specialChars = /[^\w\s-_.]/gi;
    if (specialChars.test(trimmedSearchTerm)) {
      setMessage('Special characters not allowed.');
      return;
    }
    if (trimmedSearchTerm !== '') {

      try {
        setConfirmation('Searching...');
        const response = await axios.post(`${url}/query`, {
          term: searchTerm,
        }, {
          headers: {
            'Content-Type': 'application/json',
          },
        });

        const data = response.data.map(result => result.map(item => JSON.parse(item)));
        console.log(data);
        setConfirmation('');
        renderResults(data);
        navigate('/results');
      } catch (error) {
        console.error("Error making POST request:", error);
        setMessage("Error occured while searching. Please try again.");
        setConfirmation('');
      }
    }
  };

  return (

    <div className="searchBar">
      <div id="bar">
        <InputGroup>
          <Form.Control onChange={handleSearchTermChange} />
          &nbsp;
          <Button variant="dark" onClick={handleSubmit}>
            Search
          </Button>
        </InputGroup>
      </div>
      <p style={{ textAlign: 'center' }}>{confirmation}</p>
      <p style={{ color: 'red', textAlign: 'center' }}>{message}</p>
    </div>
  );
}

export default SearchBar;
