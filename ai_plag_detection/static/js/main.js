function detectAI() {
    const inputText = document.getElementById('input-text').value;

    if (inputText.length > 0) {
        console.log("Sending Text:", inputText);

        const formData = new FormData();
        formData.append('text', inputText);

        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(result => {
            console.log("Server Response:", result);
        })
        .catch(err => console.error("AJAX Error:", err));
    }
}
