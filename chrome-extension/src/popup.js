const highlightContentButton = document.getElementById('highlightContent');

highlightContentButton.onclick = function(_element) {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        chrome.tabs.sendMessage(tabs[0].id, {text: "sendDocumentRepresentation"});

        // disable button
        highlightContentButton.textContent = "Working...";
        highlightContentButton.disabled = true;
    });
};

chrome.runtime.onMessage.addListener(function (msg, sender, _sendResponse) {
    if (msg.text === 'enableButton') {
        highlightContentButton.textContent = "Highlight content";
        highlightContentButton.disabled = false;
    }
});
