document.addEventListener('DOMContentLoaded', () => {

    const actionButton = document.getElementById('action-button');
    const actionResultDiv = document.getElementById('action-result');
    const actionStatusDiv = document.getElementById('action-status');
    const faceDataDiv = document.getElementById('face-data');

    // 1. Handle Button Click
    actionButton.addEventListener('click', () => {
        // Send a POST request to the server to toggle the state
        fetch('/toggle_action', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                // The server toggled, update UI immediately
                updateButton(data.is_recording, 0);
            });
    });

    // 2. Poll for Data Updates
    // This runs every 1 second to get the latest data from the server
    setInterval(updateData, 1000);

    function updateData() {
        fetch('/get_data')
            .then(response => response.json())
            .then(data => {
                // Update button text and color
                updateButton(data.is_recording, data.keyframe_count);

                // Update action analysis result
                actionResultDiv.textContent = data.action_result || "Waiting for analysis...";

                // Update live face data
                updateFaceData(data.live_faces);
            });
    }

    function updateButton(isRecording, keyframe_count) {
        if (isRecording) {
            actionButton.textContent = `STOP Recording (Frames: ${keyframe_count})`;
            actionButton.className = 'stop';
        } else {
            actionButton.textContent = 'Analyse Action';
            actionButton.className = 'start';
        }
    }

    function updateFaceData(faces) {
        if (faces.length === 0) {
            faceDataDiv.textContent = 'No faces detected.';
            return;
        }

        // Clear previous entries
        faceDataDiv.innerHTML = '';

        faces.forEach(face => {
            const faceEl = document.createElement('div');
            faceEl.className = 'face-item';
            
            let analysisText = face.analysis ? face.analysis : "Waiting for analysis...";
            if (face.name === 'Unknown') {
                analysisText = "<i>(Analysis only for known faces)</i>";
            }

            faceEl.innerHTML = `
                <strong>${face.name} (${face.confidence}%)</strong>
                <div>${analysisText}</div>
            `;
            faceDataDiv.appendChild(faceEl);
        });
    }

});