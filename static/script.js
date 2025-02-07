document.addEventListener("DOMContentLoaded", function () {
    // Handle X-ray Image Upload
    document.getElementById("image-form").addEventListener("submit", function (event) {
        event.preventDefault();
        let formData = new FormData();
        let fileInput = document.getElementById("xray");
        
        if (fileInput.files.length === 0) {
            alert("Please upload an X-ray image.");
            return;
        }

        formData.append("xray", fileInput.files[0]);

        fetch("/upload_xray", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            alert("X-ray uploaded successfully!");
            console.log(data);
        })
        .catch(error => console.error("Error:", error));
    });

    // Handle Clinical Data Submission
    document.getElementById("clinical-form").addEventListener("submit", function (event) {
        event.preventDefault();
        
        let clinicalData = {
            age: document.getElementById("age").value,
            height: document.getElementById("height").value,
            weight: document.getElementById("weight").value,
            bmi: document.getElementById("bmi").value,
            frequent_pain: document.getElementById("frequent_pain").value,
            surgery: document.getElementById("surgery").value,
            risk: document.getElementById("risk").value,
            sxkoa: document.getElementById("sxkoa").value,
            swelling: document.getElementById("swelling").value,
            bending_fully: document.getElementById("bending_fully").value,
            symptomatic: document.getElementById("symptomatic").value,
            crepitus: document.getElementById("crepitus").value,
            koos_pain_score: document.getElementById("koos_pain_score").value,
            osteophytes_y: document.getElementById("osteophytes_y").value,
            jsn_y: document.getElementById("jsn_y").value,
            osfl: document.getElementById("osfl").value,
            scfl: document.getElementById("scfl").value,
            cyfl: document.getElementById("cyfl").value
        };

        fetch("/predict_clinical", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(clinicalData)
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerHTML = `<h2>Prediction Result</h2><p>${data.result}</p>`;
        })
        .catch(error => console.error("Error:", error));
    });
});