function analyzeSentiment() {
    // Get the YouTube URL from the input
    const youtubeUrl = document.getElementById('youtubeUrl').value;

    // Make an AJAX request to the backend for sentiment analysis
    const backendUrl = '/analyze';
    const data = { url: youtubeUrl };

    fetch(backendUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(results => {
        // Clear previous results
        document.getElementById('result').innerHTML = '';

        // Create a table element
        const table = document.createElement('table');
        table.border = '1';

        // Add a header row to the table
        const headerRow = table.insertRow();
        const commentHeader = headerRow.insertCell(0);
        commentHeader.innerHTML = '<b>Comment</b>';

        const sentimentHeader = headerRow.insertCell(1);
        sentimentHeader.innerHTML = '<b>Sentiment</b>';

        // Add rows for each comment and sentiment
        results.forEach(result => {
            const row = table.insertRow();
            const commentCell = row.insertCell(0);
            commentCell.innerHTML = result.comment;

            const sentimentCell = row.insertCell(1);
            sentimentCell.innerHTML = result.sentiment;
        });

        // Append the table to the result div
        document.getElementById('result').appendChild(table);
    })
    .catch(error => {
        console.error('Error:', error);
        // Handle errors gracefully, e.g., display an error message to the user
    });
}
