import React, { useState, useEffect } from 'react';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';

import { DataGrid } from '@mui/x-data-grid';
import {Box, Typography, Button } from '@mui/material';

import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'
import dayjs from 'dayjs';

import { Routes, Route, useNavigate, BrowserRouter  } from "react-router-dom";

import DialogTitle from '@mui/material/DialogTitle';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';

import ArrowBackIcon from '@mui/icons-material/ArrowBack';

// import Grid2 from '@mui/material/Unstable_Grid2'; // Grid version 2
import { makeStyles } from '@mui/styles';
import ReactWordcloud from 'react-wordcloud';


function DateSelector(allowedDates, selectedDate, setSelectedDate) {
  const filterDates = date => {
    return !allowedDates.some(allowedDate => date.isSame(allowedDate, 'day'));
  };
  return (
    <div>
      <DatePicker format='DD/MM/YYYY' value={dayjs(selectedDate)} onChange={date => setSelectedDate(date.toDate())} shouldDisableDate={filterDates}/>
    </div>
  );
}

// function displayDate(date) {
//   if (date === null) {
//     return '';
//   }
//   return date.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' });
// }

function TrendingKeywords(props) {
  // console.log(setSelectedRow)
  const navigate = useNavigate();
  const [selectedDate, setSelectedDate] = useState(props.previousDate);
  const [keywords, setKeywords] = useState([]);

  const [allowedDates, setAllowedDates] = useState([]);


  const handleRowClick = (params) => {
    props.setSelectedRow(params.row);
    props.setPreviousDate(selectedDate)
    navigate("/details");
  };

  useEffect(() => {
    if (allowedDates.length === 0) {
      fetch('/available_dates')
      .then(response => response.json())
      .then(data => {
        const dates = data.map(item => new dayjs(item.date));
        setAllowedDates(dates);
      })
      
    }
    if (selectedDate === null) {
      fetch('/latest_date')
        .then(response => response.json())
        // get the date part of the result
        .then(data => {
          setSelectedDate(new Date(data[0].latest_date))
        })
        .catch(error => console.log(error));
    }
    
    
  }, []);

  useEffect(() => {
    if (selectedDate === null) {
      return;
    }
    fetch(`/trending?date=${selectedDate.toISOString().slice(0, 10)}`)
      .then(response => response.json())
      .then(data => {
        setKeywords(data);
      });
  }, [selectedDate]);

  
  const columns = [
    { field: 'display_order', headerName: '', width: 50},
    { field: 'keyword', headerName: 'Keyword', width: 250 },
    { field: 'positive_score', headerName: 'Positive Score', width: 140 },
    { field: 'post_collected', headerName: 'Post Collected', width: 140 },
  ];

  return (
    <div>
      <h1>Trending Keywords</h1>
      <h2>Date: {DateSelector(allowedDates, selectedDate, setSelectedDate)}</h2>
      

      <div style={{ height: 700, width: 600 }}>
          <DataGrid rows={keywords} columns={columns} hideFooter={true} onRowClick={handleRowClick}/>
      </div>
      
    </div>
  );
}

const useStyles = makeStyles((theme) => ({
  container: {
    display: 'flex',
    justifyContent: 'space-between',
  },
}));

function DetailsPage({ rowData }) {
  const classes = useStyles();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const [opinions, setOpinions] = useState([]);
  const [opinions1, setOpinions1] = useState([]);
  const [wordCloudData, setWordCloudData] = useState([]);
  const [wordCloudData1, setWordCloudData1] = useState([]);

  const [similarOpinion, setSimilarOpinion] = useState([]);

  useEffect(() => {
    // get by keyword_id
    // alert(rowData.id)
    if (rowData === null) {
      return;
    }
    fetch(`/opinions?keyword_id=${rowData.id}`)
      .then(response => response.json())
      .then(data => {
        let data0 = data.filter(item => item.class === 0);
        let data1 = data.filter(item => item.class === 1);
        setOpinions(data0);
        setOpinions1(data1);
        // console.log(data.map(item => ({text: item.text , value: item.agg_score})))
        setWordCloudData(data0.map(item => ({text: item.text , value: item.agg_score})));
        setWordCloudData1(data1.map(item => ({text: item.text , value: item.agg_score})));
      });
  }, []);
  

  const columns = [
    { field: 'text', headerName: 'Opinion', width: 1100},
    { field: 'posts', headerName: 'Posts', width: 100 },
    { field: 'likes', headerName: 'Likes', width: 100 },
    { field: 'agg_score', headerName: 'Score', width: 100 },
  ];

  const relatedColumns = [
    { field: 'similar_opinion', headerName: 'Similar Opinion', width: 450},
    { field: 'similarity', headerName: 'Similarity', width: 100 },
  ];

  function handleOpen(params) {
    fetch(`/similar_opinions?opinion_id=${params.row.id}`)
      .then(response => response.json())
      .then(data => {
        setSimilarOpinion(data);
      });
    setOpen(true);
  }


  const options = {
    rotations: 2,
    rotationAngles: [0, 5],
    fontSizes: [20, 40],
    fontFamily: "impact",
    scale: "sqrt",
    spiral: "archimedean",
    transitionDuration: 1000

  };
  const size = [1400, 600];

  return (
    // create back button
    <div >
      <Box className={classes.container} marginTop={2} marginBottom={2}>
      <Button onClick={() => navigate(-1)} startIcon={<ArrowBackIcon/>  }>
            Back
          </Button>
          <Typography variant="h3" color="primary">{rowData.keyword}</Typography>
          <Typography variant="h3" color="primary">
            <span style={{color: 'green'}}>{rowData.positive_score}</span>
            </Typography>
      </Box>

      <Typography variant='h4'>Similar Opinions</Typography>
      <ReactWordcloud 
        words={wordCloudData} 
        options={options}
        size={size}
        maxWords={900}
      />
      <div style={{ height: 400, width: '100%' }}>
          <DataGrid rows={opinions} columns={columns} hideFooter={true} onRowClick={handleOpen}/>
      </div>

      <Typography variant='h4'>Entailed Opinions</Typography>
      <ReactWordcloud 
        words={wordCloudData1} 
        options={options}
        size={size}
        maxWords={900}
      />
      <div style={{ height: 400, width: '100%' }}>
          <DataGrid rows={opinions1} columns={columns} hideFooter={true} onRowClick={handleOpen}/>
      </div>


      <Dialog open={open} onClose={() => setOpen(false)}>
        <DialogTitle>Similar Opinions</DialogTitle>
        <DialogContent>
          <DialogContentText>
            <div style={{ height: 650, width: '100%' }}>
              <DataGrid rows={similarOpinion} columns={relatedColumns}/>
          </div>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}

function App() {
  const [selectedRow, setSelectedRow] = useState(null);

  const [previousDate, setPreviousDate] = useState(null);

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      margin={3}
    >
      <Routes>
      <Route path="/" element={<TrendingKeywords setSelectedRow={setSelectedRow} previousDate={previousDate} setPreviousDate={setPreviousDate} />}/>
      <Route path="/details" element={<DetailsPage rowData={selectedRow} />} />
      </Routes>
    </Box>
    </LocalizationProvider>
  );
}

function Root() {
  return (
    <BrowserRouter>
      <App />
    </BrowserRouter>
  );
}
export default Root;
