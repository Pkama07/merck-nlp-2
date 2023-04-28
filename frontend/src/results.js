import React, { useState, useEffect, useRef } from 'react';
import Button from 'react-bootstrap/Button';
import AWS from 'aws-sdk';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import Upload from './upload';
import Logo from './logo';
import { Link } from 'react-router-dom';
import SearchBar from './searchBar';

const accessKeyId = process.env.REACT_APP_AWS_ACCESS_KEY;
const secretAccessKey = process.env.REACT_APP_AWS_SECRET_ACCESS_KEY;
const s3_bucket = process.env.REACT_APP_AWS_BUCKET;

// Set the region
AWS.config.update({ region: "us-east-1" });

// Set the credentials
AWS.config.update({
  accessKeyId: accessKeyId,
  secretAccessKey: secretAccessKey,
});

export default function Results({ resultsData, renderResults }) {
  const [results, setResults] = useState([]);
  const resultsRef = useRef(null);
  const [downloadStatus, setDownloadStatus] = useState(null);

  useEffect(() => {
    if (Array.isArray(resultsData)) {
      console.log("is an array");
      const flattenedResults = resultsData.flatMap(arr => arr.filter(item => typeof item === 'object'));
      setResults(flattenedResults);
    } else {
      setResults([]);
    }
  }, [resultsData]);

  useEffect(() => {
    console.log("Results state:", results);
  }, [results]);


  function generatePresignedUrl(bpNumber, fileType) {
    const s3 = new AWS.S3();
    const params = {
      Bucket: s3_bucket,
      Key: `${bpNumber}.` + (fileType === 'pdf' ? 'pdf' : 'xlsx'),
      Expires: 300, // URL expiration time in seconds
    };

    return s3.getSignedUrl('getObject', params);
  }

  async function downloadAndAddToZip(url, filename, zip) {
    console.log('Downloading:', url, 'Filename:', filename);
    const response = await fetch(url);
    const blob = await response.blob();
    return zip.file(filename, blob);
  }

  async function handleDownload() {
    setDownloadStatus('downloading');
    try {
      const selectedBps = [...new Set(results.filter(bp => bp.selected).map(bp => bp["BP_Number: "]))]; // Remove duplicates
      if (selectedBps.length > 0) {
        for (const bpNumber of selectedBps) {
          const pdfUrl = generatePresignedUrl(bpNumber, 'pdf');
          const excelUrl = generatePresignedUrl(bpNumber, 'excel');

          const zip = new JSZip();

          await Promise.all([
            downloadAndAddToZip(pdfUrl, `${bpNumber}-pdf.pdf`, zip),
            downloadAndAddToZip(excelUrl, `${bpNumber}-excel.xlsx`, zip),
          ]);

          const zipBlob = await zip.generateAsync({ type: 'blob' });
          saveAs(zipBlob, `${bpNumber}-files.zip`);
          await new Promise(resolve => setTimeout(resolve, 100)); // Wait time in milliseconds before the next download
        }
        setDownloadStatus('completed');
        setTimeout(() => {
          setDownloadStatus('');
        }, 2500);
      } else {
        setDownloadStatus('none');
        setTimeout(() => {
          setDownloadStatus('');
        }, 2500);
      }

    } catch (error) {
      console.error('Error during download:', error);
      setDownloadStatus('error');
      setTimeout(() => {
        setDownloadStatus('');
      }, 2500);
    }
  }



  function handleCheckboxChange(bpNumber) {
    setResults(prevResults => {
      // Determine the new selection state for all items with the same BP number
      const targetItem = prevResults.find(bp => bp && bp["BP_Number: "] === bpNumber);
      if (!targetItem) return prevResults; // If no item found, return the previous state

      const targetSelectedState = !targetItem.selected;

      return prevResults.map(bp => {
        if (bp && bp["BP_Number: "] === bpNumber) {
          return {
            ...bp,
            selected: targetSelectedState
          };
        } else {
          return bp;
        }
      });
    });
  }

  let downloadStatusMessage;

  if (downloadStatus === 'downloading') {
    downloadStatusMessage = <p>Downloading files, please wait...</p>;
  } else if (downloadStatus === 'completed') {
    downloadStatusMessage = <p>Download completed.</p>;
  } else if (downloadStatus === 'error') {
    downloadStatusMessage = <p>Error: Download failed. Please try again.</p>;
  } else if (downloadStatus === 'none') {
    downloadStatusMessage = <p>No documents selected.</p>
  } else {
    downloadStatusMessage = <p></p>
  }

  if (results.length > 0) {
    return (
      <div className='topBar'>
        <div className='titleBar' style={{ display: "flex", justifyContent: "space-between" }}>
          <div id='logo'>
            <Link to="/">
              <Logo />
            </Link>
          </div>
          <div id='search'>
            <SearchBar renderResults={renderResults} />
          </div>
          <div id='upload' style={{ marginRight: "2vw" }}>
            <Upload />
          </div>
        </div>

        <div ref={resultsRef} id='results'>
          <div className='results-box'>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div>
                <h1>Results</h1>
              </div>
              <div>
                <Button
                  onClick={handleDownload}
                  style={{ marginRight: "2vw" }}
                  variant="dark"
                >
                  Download Selected Documents
                </Button>
              </div>
            </div>
            <div style={{ textAlign: 'right', marginRight: '2vw' }}>
              {downloadStatusMessage}
            </div>
            {results.map((bp, index) => (
              <div key={`${bp["BP_Number: "]}-${index}`}>
                <h2>{bp["BP_Number: "]}</h2>
                <table>
                  <thead>
                    <tr>
                      <th>Select</th>
                      <th>MK Number</th>
                      <th>Species</th>
                      <th>Matrix</th>
                      <th>Extraction Method</th>
                      <th>Internal Standard</th>
                      <th>Chromatography</th>

                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <input
                          style={{ height: "24px", width: "24px", marginRight: "6vw" }}
                          type="checkbox"
                          checked={bp.selected}
                          onChange={() => handleCheckboxChange(bp["BP_Number: "])}
                        />
                      </td>
                      <td>{bp["MK_Number: "]}</td>
                      <td>{bp["Species: "]}</td>
                      <td>{bp["Matrix: "]}</td>
                      <td>{bp["Extraction_Method: "]}</td>
                      <td>{bp.Internal_Standard}</td>
                      <td>{bp["Chromatography: "]}</td>
                    </tr>
                  </tbody>
                </table>
                <hr></hr>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  } else {
    return (
      <div className='topBar'>
        <div className='titleBar' style={{ display: "flex", justifyContent: "space-between" }}>
          <div id='logo'>
            <Link to="/">
              <Logo />
            </Link>
          </div>
          <div id='search'>
            <SearchBar renderResults={renderResults} />
          </div>
          <div id='upload' style={{ marginRight: "2vw" }}>
            <Upload />
          </div>
        </div>
        <div ref={resultsRef} id='results'>
          <div className='results-box'>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <div>
                <h3> No Results Found</h3>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

}
