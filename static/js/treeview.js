function openTreeModal(treeData) {
    $('#treeview').jstree({
        'core': {
            'data': treeData
        }
    });

    $("#treeModal").dialog({
        modal: true,
        width: 400,
        height: 300
    });
}

$("#openModalBtn").on("click", function() {
    $.ajax({
        url: '/get_directory_tree',
        type: 'GET',
        dataType: 'json',
        success: function(data) {
            openTreeModal(data);
        },
        error: function(xhr, status, error) {
            console.error(status + ': ' + error);
        }
    });
});