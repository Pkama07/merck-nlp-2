import React, { useState, useRef, useEffect } from 'react';
import DialogContentText from '@mui/material/DialogContentText';
import LinearProgress from '@mui/material/LinearProgress';
import './upload.css';
import Dialog from '@mui/material/Dialog';
import LogoSmall from './logoSmall';
import DialogContent from '@mui/material/DialogContent';
import Button from 'react-bootstrap/Button';
import { DialogActions, DialogTitle } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import TextareaAutosize from '@mui/material/TextareaAutosize';

function Upload() {
  const [showCloseConfirm, setShowCloseConfirm] = useState(false);

  const [progress, setProgress] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [message, setMessage] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const intervalRef = useRef(null);

  const [reviewMessage, setReviewMessage] = useState('');
  const [reviewConfirmation, setReviewConfirmation] = useState('');

  const [expanded, setExpanded] = useState(false);
  const [expandedResults, setExpandedResults] = useState(false);
  //state variable for json response from backend
  const [parsed, setParsed] = useState({});
  const [updatedValues, setUpdatedValues] = useState({});

  const [acceptedFile, setAcceptedFile] = useState(null);
  const timers = useRef([]);
  const url = process.env.REACT_APP_API_ENDPOINT;

  useEffect(() => {
    const beforeUnloadHandler = (e) => {
      if (intervalRef.current) {
        e.preventDefault();
        e.returnValue = 'Ongoing operation. Are you sure you want to leave?';
      }
    };
  
    window.addEventListener('beforeunload', beforeUnloadHandler);
    return () => {
      window.removeEventListener('beforeunload', beforeUnloadHandler);
    };
  }, []);
  

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  const handleCloseConfirm = () => {
    setShowCloseConfirm(false);
  };

  const handleTextareaChange = (key, e) => {
    setUpdatedValues((prevState) => ({ ...prevState, [key]: e.target.value }));
  };

  const mergeUpdatedValues = (updatedValues) => {
    // Deep copy of original data to avoid directly mutating the state
    const mergedData = JSON.parse(JSON.stringify(parsed));

    // Update the merged data with changes made by the user
    Object.keys(updatedValues).forEach((key) => {
      mergedData[key] = updatedValues[key];
    });

    return mergedData;
  };


  const handleClickOpen = () => {
    setExpanded(true);
  };

  const handleClose = () => {
    if (intervalRef.current) {
      setShowCloseConfirm(true);
    } else {
      cancelProgress();
      setProgress(0);
      setShowProgress(false);
      setMessage('');
      setConfirmation('');
      setExpanded(false);
      setAcceptedFile(null);
    }
  };

  const handleConfirmClose = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    handleCloseConfirm();
    handleClose();
  };



  const handleResultsClose = async () => {
    setReviewConfirmation('Saving data...');
    // Send the updated data to the backend
    try {
      const mergedData = mergeUpdatedValues(updatedValues);
      console.log('Sending data to server:', mergedData);
      await axios.post(`${url}/save`, { 'fields': mergedData });
      setReviewConfirmation('Data saved successfully.');
    } catch (error) {
      setReviewMessage('An error occured while saving the data.');
    }

    setTimeout(() => {
      setExpandedResults(false);
      setReviewConfirmation('');
      setReviewMessage('');
      handleClose();
    }, 2000);

  };

  const handleResultsOpen = () => {
    setExpandedResults(true);
  };

  const cancelProgress = () => {
    timers.current.forEach((timer) => {
      clearTimeout(timer);
    });
    timers.current = [];
  };

  const progressiveDelay = (i, initial, step, scaleFactor) => {
    return Math.pow(i - initial, scaleFactor) * step;
  };

  const upload = async () => {
    if (!acceptedFile) {
      setMessage('Please add a document to upload.');
      return;
    }

    setMessage('');
    setConfirmation('');
    const form = new FormData();
    form.append('file', acceptedFile);
    setShowProgress(true);

    const checkParsedData = async (filename) => {
      for (let i = 20; i < 98; i++) {
        const timer = setTimeout(() => {
          setProgress(i);
        }, progressiveDelay(i, 20, 100, 1.53));
        timers.current.push(timer);
      }
      intervalRef.current = setInterval(async () => {
        try {
          const response = await axios.post(`${url}/check_doc_status`, { doc_name: filename });
          console.log(response);

          if (response.data.status === 'FINISHED') {
            cancelProgress();
            setProgress(100);
            setConfirmation('File Upload Complete.');
            clearInterval(intervalRef.current);
            intervalRef.current = null;

            // Display the received data to the user for the review page
            setParsed(response.data.values);
            console.log(response.data.values);
            handleResultsOpen();
          } else if (response.data.status === 'DNE' || response.data.status === 'ERROR') {
            clearInterval(intervalRef.current);
            intervalRef.current = null;

            setMessage('An error occurred while processing the file. Please try again.');
            setConfirmation('');
            cancelProgress();
            setProgress(0);
            setShowProgress(false);
          }
        } catch (error) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;

          setMessage('An error occurred while parsing the data. Please try again.');
          cancelProgress();
          setProgress(0);
          setShowProgress(false);
          setConfirmation('');
          console.log(error);
        }
      }, 10000); // Check every 10 seconds
    };

    try {
      const response = await axios.post(`${url}/upload`, form, {
        headers: { 'Content-Type': 'undefined' },
        timeout: 600000,
        onUploadProgress: (progressEvent) => {
          const { loaded, total } = progressEvent;
          const percentageProgress = Math.round((loaded * 20) / total);
          setConfirmation('Uploading...');
          setProgress(percentageProgress);
        },
      });

      if (response.data.status === 'SUCCESS') {
        checkParsedData(acceptedFile.name);
        setConfirmation('Parsing...');
        console.log(response);
        console.log(response.data.status);
      } else {
        setMessage('An error occurred while uploading the file. Please try again.');
        setConfirmation('');
        cancelProgress();
        setProgress(0);
        setShowProgress(false);
      }


    } catch (error) {
      setMessage('An error occurred while uploading the file. Please try again.');
      console.log(error);
      setConfirmation('');
      cancelProgress();
      setProgress(0);
      setShowProgress(false);

    }
  };


  const buildResults = () => {
    const results = [];
    if (parsed !== null) {
      for (const key in parsed) {
        const val = parsed[key];
        if (typeof val === 'string' || val === null) {
          results.push(
            <tr key={key}>
              <td>{key}</td>
              <td>
                <TextareaAutosize
                  defaultValue={val || ''}
                  style={{ border: 'none' }}
                  onChange={(e) => handleTextareaChange(key, e)}
                />
              </td>
            </tr>
          );
        }
      }
    }
    return results;
  };


  const { getRootProps, getInputProps } = useDropzone({
    accept: 'application/pdf',
    onDrop: (acceptedFiles) => {
      const file = acceptedFiles[0];
      const fileNamePattern = /^BP-\d{4}\.pdf$/;
      if (fileNamePattern.test(file.name)) {
        setAcceptedFile(file);
        setMessage('');
      } else {
        setMessage('Invalid file name. Please upload a file in the format: BP-####.pdf');
      }
    },
  });

  return (
    <div>
      <Button variant="dark" onClick={handleClickOpen} >
        Upload PDF
      </Button>

      <Dialog open={expanded} onClose={handleClose} fullWidth maxWidth="sm">
        <DialogTitle>
          <div id="logoSmall">
            <LogoSmall />
          </div>
        </DialogTitle>
        <DialogContent>
          <section>
            <div id="content" {...getRootProps()}>
              <input {...getInputProps()} />
              <div className="drag-area">
                <div className="icon">
                  <i className="fas fa-cloud-upload-alt"></i>
                </div>
                <header>Drag & Drop to Upload PDF</header>
                <span>or</span>
                <button>Browse PDF</button>
              </div>
            </div>
            <aside>
              <ul>
                {acceptedFile && (
                  <li key={acceptedFile.path}>
                    {acceptedFile.path} -{' '}
                    {(acceptedFile.size / 1000000).toFixed(2)} MB
                  </li>
                )}
              </ul>

            </aside>
          </section>
          {showProgress && <LinearProgress variant="determinate" value={progress} />}
          <p style={{ textAlign: 'center' }}>{confirmation}</p>
          <p style={{ color: 'red', textAlign: 'center' }}>{message}</p>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} color="primary">
            Cancel
          </Button>
          <Button onClick={upload} color="primary">
            Upload
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog open={expandedResults} onClose={handleClose} fullWidth maxWidth="sm">
        <DialogTitle>
          <div id='logoSmall'>
            <LogoSmall />
          </div>
        </DialogTitle>
        <div className="resultsHead">
          <header>Document Review</header>
          <h3>Please Confirm the Parsed Information Below</h3>
          <h5>Click any field to edit</h5>
        </div>
        <div className="resultsList">
          <table>
            <tbody>{buildResults()}</tbody>
          </table>
        </div>
        <div className="resultsBtn">
          <button onClick={handleResultsClose}>Save</button>
          <p style={{ textAlign: 'center' }}>{reviewConfirmation}</p>
          <p style={{ color: 'red', textAlign: 'center' }}>{reviewMessage}</p>
        </div>
      </Dialog>
      <Dialog
        open={showCloseConfirm}
        onClose={handleCloseConfirm}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">{"Confirm Close"}</DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            Are you sure you want to cancel the PDF upload?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseConfirm} color="primary">
            No
          </Button>
          <Button onClick={handleConfirmClose} color="primary" autoFocus>
            Yes
          </Button>
        </DialogActions>
      </Dialog>

    </div>
  )

}

export default Upload;