const express = require('express');
const mysql = require('mysql');

const app = express();

const fs = require('fs');

const data = fs.readFileSync('../../cred/sql.txt');
const lines = data.toString('ascii').trim().split('\n');
const account = lines[0].trim();
const password = lines[1].trim();

// Create MySQL connection
const connection = mysql.createConnection({
  host: 'sophia.cs.hku.hk',
  user: account,
  password: password,
  database: account
});

app.get('/latest_date', (req, res) => {
  connection.query('SELECT MAX(DATE(date)) AS latest_date FROM FYP_Keywords', (error, results, fields) => {
    if (error) throw error;
    // console.log(results)
    res.json(results);
  });
});

app.get('/available_dates', (req, res) => {
  connection.query('SELECT DISTINCT date FROM FYP_Keywords', (error, results, fields) => {
    if (error) throw error;
    res.json(results);
  });
});

// Define API endpoint to retrieve trending keywords for a given date
app.get('/trending', (req, res) => {
  // console.log(req.query.date)
  connection.query('SELECT * FROM FYP_Keywords WHERE DATE(date) = ? ORDER BY display_order', [req.query.date], (error, results, fields) => {
    if (error) throw error;
    // console.log(results)
    res.json(results);
  });
});

app.get('/opinions', (req, res) => {
  // console.log(req.query.keyword)
  connection.query('SELECT * FROM FYP_Opinion WHERE keyword_id = ? ORDER BY agg_score DESC', [req.query.keyword_id], (error, results, fields) => {
    if (error) throw error;
    // console.log(results)
    res.json(results);
  });
});

app.get('/similar_opinions', (req, res) => {
  // console.log("-----")
  // console.log(req.query.opinion_id)
  connection.query('SELECT * FROM FYP_Similar WHERE opinion_id = ? ORDER BY similarity DESC', [req.query.opinion_id], (error, results, fields) => {
    if (error) throw error;
    // console.log(results)
    res.json(results);
  });
});
  

// Start server
app.listen(3001, () => {
  console.log('Server started on port 3001');
});
