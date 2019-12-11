const extractContentButton = document.getElementById('extractContent');

extractContentButton.onclick = function(_element) {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs){
        chrome.tabs.sendMessage(tabs[0].id, {text: "sendDocumentRepresentation"});  
    });
};
