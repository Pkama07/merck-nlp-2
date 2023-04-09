import React, { useState } from 'react';
import './searchBar.css';
import Dropdown from 'react-bootstrap/Dropdown';
import DropdownButton from 'react-bootstrap/DropdownButton';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';

function SearchBar({renderResults}) {
    const [searchTerm, setSearchTerm] = useState('');

    const handleSearchTermChange = (event) => {
        setSearchTerm(event.target.value);
    }

    const handleSubmit = (event) => {
        event.preventDefault();
        // following link will need to be replaced with url of deployed API
        fetch("http://0.0.0.0:8080/searchDB/" + searchTerm + "+Movie+title")
        .then((response) => response.json())
        .then((data) => console.log(data));
        // here we want to send a get request to the API to return all of the info from the database
        //alert('A search was submitted\n\nSearch Term: ' + searchTerm + '\n\nSearch Category: ' + searchCategory);
        //make a call to the backend to search for this term
    }

    return (
        <div className="searchBar">
            <div id='bar'>
                <InputGroup>
                    <Form.Control onChange={handleSearchTermChange}/>
                    &nbsp;
                    <div id='submit'>
                        <input id="submit-btn" type="submit" value="Search" onClick={handleSubmit} />
                    </div>
                </InputGroup>
            </div>
        </div>

    );
}

export default SearchBar;