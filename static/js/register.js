let mediaRecorder;
let audioChunks = [];
let audioBlob;


const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const submitBtn = document.getElementById('submitBtn');

// =========================================
// START RECORDING
// =========================================

startBtn.addEventListener('click', async () => {

    const stream = await navigator.mediaDevices.getUserMedia({
        audio: true
    });

    mediaRecorder = new MediaRecorder(stream, {
    mimeType: 'audio/webm'
    });

    audioChunks = [];

    mediaRecorder.start();

    startBtn.disabled = true;
    stopBtn.disabled = false;

     mediaRecorder.addEventListener('dataavailable', event => {
        audioChunks.push(event.data);
    });


    mediaRecorder.addEventListener('stop', () => {

        audioBlob = new Blob(audioChunks, {
            type: 'audio/wav'
        });

        const audioUrl = URL.createObjectURL(audioBlob);

        document.getElementById('audioPlayback').src = audioUrl;
    });
});

// =========================================
// STOP RECORDING
// =========================================

stopBtn.addEventListener('click', () => {

    mediaRecorder.stop();

    startBtn.disabled = false;
    stopBtn.disabled = true;
});
// =========================================
// SUBMIT REGISTRATION
// =========================================

submitBtn.addEventListener('click', async () => {

    const name = document.getElementById('name').value;

    const userId = document.getElementById('userId').value;

    const model = document.getElementById('modelSelect').value;

    const resultBox = document.getElementById('result');


    if (!audioBlob) {
        alert('Please record audio first');
        return;
    }

     const formData = new FormData();

    formData.append('audio', audioBlob, 'recording.wav');

    formData.append('name', name);

    formData.append('user_id', userId);

    formData.append('model', model);


    resultBox.innerHTML = 'Registering...';


    const response = await fetch('/register', {
        method: 'POST',
        body: formData
    });

     const data = await response.json();


    resultBox.innerHTML = `
        <b>Status:</b> ${data.status}<br>
        <b>User:</b> ${data.name}<br>
        <b>Model:</b> ${data.model}<br>
        <b>Message:</b> ${data.message}
    `;
});