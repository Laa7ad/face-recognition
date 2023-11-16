$(document).ready(function() {
    $("#trainButton").on("click", function() {
        var source = new EventSource('/train_model');
        
        $('#trainingModal').modal('show');

        source.onmessage = function(event) {
            var response = JSON.parse(event.data);
            if (response && response.status === 'success') {
                $("#trainingStatus").text(response.message);
                $("#trainingStatus").append('<br>Train Accuracy: ' + response.accuracy_train);
                $("#trainingStatus").append('<br>Test Accuracy: ' + response.accuracy_test);
                source.close();

                $('#trainingModal').modal('hide');
                setTimeout(function() {
                    window.location.reload();
                }, 30000);
            } else if (response && response.status === 'error') {
                $("#trainingStatus").text("Error occurred during training!");
                source.close();

                $('#trainingModal').modal('hide');
            }
        };
    });
});
