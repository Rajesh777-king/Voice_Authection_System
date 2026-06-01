let mediaRecorder;
let audioChunks = [];

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const verifyBtn = document.getElementById("verifyBtn");

const audioPlayback =
      document.getElementById("audioPlayback");

startBtn.onclick = async () => {

    const stream =
        await navigator.mediaDevices.getUserMedia({
            audio: true
        });

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.start();

    audioChunks = [];

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {

        const audioBlob =
            new Blob(audioChunks,
            { type: 'audio/webm' });

        audioPlayback.src =
            URL.createObjectURL(audioBlob);

        window.recordedAudio = audioBlob;
    };
};

stopBtn.onclick = () => {
    mediaRecorder.stop();
};

verifyBtn.onclick = async () => {

    const formData = new FormData();

    formData.append(
        "audio",
        window.recordedAudio,
        "verify.webm"
    );

    formData.append(
        "name",
        document.getElementById("name").value
    );

    formData.append(
        "user_id",
        document.getElementById("user_id").value
    );

    formData.append(
        "model",
        document.getElementById("model").value
    );

    const response = await fetch("/verify", {
        method: "POST",
        body: formData
    });

    const result = await response.json();

    document.getElementById("result").innerHTML = `
        <h2>Status: ${result.status}</h2>
        <h2>User: ${result.name}</h2>
        <h2>Similarity: ${result.similarity}</h2>
        <h2>Message: ${result.message}</h2>
    `;
};