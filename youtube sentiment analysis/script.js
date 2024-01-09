    function analyzeSentiment() {
        // Get the YouTube URL from the input
        const youtubeUrl = document.getElementById('youtubeUrl').value;

        // Perform validation on the URL if needed

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
        .then(result => {
            // Display the sentiment analysis result
            document.getElementById('result').innerHTML = `<p>Sentiment: ${result.sentiment}</p>`;
        })
        .catch(error => {
            console.error('Error:', error);
            // Handle errors gracefully, e.g., display an error message to the user
        });
    }
