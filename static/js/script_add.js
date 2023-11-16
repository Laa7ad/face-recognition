document.addEventListener('DOMContentLoaded', function() {
    var currentDate = new Date();
    var formattedDate = currentDate.toISOString().slice(0, 10).replace(/-/g, '');
    var formattedTime = currentDate.toTimeString().slice(0, 8).replace(/:/g, '');

    document.getElementById('folder_name').value = "personne" + formattedDate + formattedTime;

    function captureFaces() {
        var folderName = document.getElementById('folder_name').value;
        var resultDiv = document.getElementById('result');

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/capture', true);
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        $('#trainingModal').modal('show');
        xhr.onload = function() {
            if (xhr.status === 200) {
                resultDiv.innerText = 'Faces captured successfully in folder: ' + folderName;
                $('#trainingModal').modal('hide');
                $('#messagemodal').modal('show');
                reloadPage()
            } else {
                resultDiv.innerText = 'Error capturing faces';
                $('#messagemodal').modal('show');
            }
        };

        xhr.send('folder_name=' + folderName);
    }

    function reloadPage() {
        setTimeout(function() {
            location.reload();
        }, 1500);
    }

    // Function to handle the 'Capture Faces' button click
    document.getElementById('captureButton').addEventListener('click', function() {
        captureFaces();
    });
});
