async function recognizeSpeaker() {

    const fileInput = document.getElementById('audioFile');

    const modelSelect = document.getElementById('modelSelect');

    const resultBox = document.getElementById('result');


    if (fileInput.files.length === 0) {
        alert('Please upload audio file');
        return;
    }


    const formData = new FormData();

    formData.append('audio', fileInput.files[0]);
     formData.append('model', modelSelect.value);


    resultBox.innerHTML = 'Processing...';


    const response = await fetch('/recognize', {
        method: 'POST',
        body: formData
    });


    const data = await response.json();


    resultBox.innerHTML = `
        <b>Status:</b> ${data.status}<br>
        <b>Model:</b> ${data.model}<br>
        <b>Speaker:</b> ${data.speaker}<br>
        <b>Confidence:</b> ${data.confidence}
    `;
}
