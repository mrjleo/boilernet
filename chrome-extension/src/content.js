var IGNORE_TAGS = new Set(
    ['head', 'iframe', 'script', 'meta', 'link', 'style', 'input', 'checkbox', 'button', 'noscript']
);

chrome.runtime.onMessage.addListener(function (msg, _sender, _sendResponse) {
    console.log('content.js received msg: ' + msg.text);
    if (msg.text === 'sendDocumentRepresentation') {
        sendDocumentRepresentation();
    } else if (msg.text === 'visualize') {
        injectPredictions(msg.predictions);
        highlightContent();
    }
});

function wrap(node, leafIndex) {
    const wrapper = document.createElement('span');
    wrapper.setAttribute('boilernet_index', leafIndex);
    node.parentNode.insertBefore(wrapper, node);
    wrapper.appendChild(node);
}

async function sendDocumentRepresentation() {
    var leafIndex = 0;
    function _helper(node, tagList) {
        const tagListNew = tagList.slice();
        tagListNew.push(node.nodeName.toLowerCase());
        const result = [];
        node.childNodes.forEach(c => {
            if (c.nodeName === '#text') {
                const text = c.textContent.trim().toLowerCase();
                if (text) {
                    result.push({text: text, tags: tagListNew});
                    // wrap the text node in a span tag that has an index
                    // we use this later to show the results in the browser
                    wrap(c, leafIndex++);
                }
            } else if (!IGNORE_TAGS.has(c.nodeName.toLowerCase())) {
                result.push(..._helper(c, tagListNew));
            }
        });
        return result;
    }
    const result = _helper(document.documentElement, []);
    chrome.runtime.sendMessage({text: "classify", documentRepresentation: result});
}

function injectPredictions(predictions) {
    const walker = document.createTreeWalker(
        document.documentElement,
        NodeFilter.SHOW_ELEMENT,
        node => {
            if (node.hasAttribute('boilernet_index')) {
                return NodeFilter.FILTER_ACCEPT;
            }
            return NodeFilter.FILTER_SKIP;
        });
    while (walker.nextNode()) {
        const index = walker.currentNode.getAttribute('boilernet_index');
        if (Math.round(predictions[index])) {
            walker.currentNode.classList.add('boilernet_content');
        }
    }
}

function highlightContent() {
    const rule = '.boilernet_content {background-color: yellow;}';
    const css = document.createElement('style');
    css.type = 'text/css';
    css.appendChild(document.createTextNode(rule));
    document.getElementsByTagName('head')[0].appendChild(css);
}
