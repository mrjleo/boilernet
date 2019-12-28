import 'babel-polyfill';
import * as tfjs from '@tensorflow/tfjs';
import tokenizer from 'wink-tokenizer';

function loadModel(fileName) {
    const url = chrome.runtime.getURL(fileName);
    return tfjs.loadLayersModel(url);
}

async function readVocab(fileName) {
    const url = chrome.runtime.getURL(fileName);
    const response = await fetch(url);
    return response.json();
}

const TOKENIZER = tokenizer();

chrome.runtime.onMessage.addListener(function (msg, sender, _sendResponse) {
    if (msg.text === 'classify') {
        classify(msg.documentRepresentation, sender.tab);
    }
});

function getInputs(documentRepresentation, words, tags) {
    // create zero buffer of the correct shape
    const numWords = Object.keys(words).length;
    const numTags = Object.keys(tags).length;
    const shape = [1, documentRepresentation.length, numWords + numTags];
    const inputs = tfjs.buffer(shape);
    
    // for each word and tag, increment the value at the corresponding index
    documentRepresentation.forEach((leaf, leafIndex) => {
        TOKENIZER.tokenize(leaf.text).forEach(token => {
            let wordIndex = words['<UNK>'];
            if (words.hasOwnProperty(token.value)) {
                wordIndex = words[token.value];
            }
            const oldVal = inputs.get(0, leafIndex, wordIndex);
            inputs.set(oldVal + 1, 0, leafIndex, wordIndex);
        });
        leaf.tags.forEach(tag => {
            let tagIndex = numWords + tags['<UNK>'];
            if (tags.hasOwnProperty(tag)) {
                tagIndex = numWords + tags[tag];
            }
            const oldVal = inputs.get(0, leafIndex, tagIndex);
            inputs.set(oldVal + 1, 0, leafIndex, tagIndex);
        });
    });
    return inputs.toTensor();
}

async function classify(documentRepresentation, tab) {
    const model = await loadModel('files/model.json');
    const words = await readVocab('files/words.json');
    const tags = await readVocab('files/tags.json');
    const inputs = getInputs(documentRepresentation, words, tags);
    const predictions = await model.predict(inputs).data();
    chrome.tabs.sendMessage(tab.id, {text: 'visualize', predictions: predictions});
    chrome.runtime.sendMessage({text: 'enableButton'});
}
