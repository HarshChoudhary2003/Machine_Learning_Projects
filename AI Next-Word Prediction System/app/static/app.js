// -----------------------------------------------------------------------------
// DOM ELEMENTS
// -----------------------------------------------------------------------------
const editorInput = document.getElementById('editor-input');
const editorBackdrop = document.getElementById('editor-backdrop');
const charCount = document.getElementById('char-count');
const wordCount = document.getElementById('word-count');

const noSuggestions = document.getElementById('no-suggestions');
const suggestionsList = document.getElementById('suggestions-list');

const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const kSlider = document.getElementById('k-slider');
const kVal = document.getElementById('k-val');
const cacheToggle = document.getElementById('cache-toggle');

const latencyVal = document.getElementById('latency-val');
const cacheHitsVal = document.getElementById('cache-hits-val');

// -----------------------------------------------------------------------------
// APP STATE
// -----------------------------------------------------------------------------
let activeSuggestion = "";
let currentSuggestions = [];
let debounceTimer = null;
let cacheHits = 0;
const DEBOUNCE_DELAY_MS = 150;

// -----------------------------------------------------------------------------
// UTILITY FUNCTIONS
// -----------------------------------------------------------------------------
function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function syncScroll() {
    editorBackdrop.scrollTop = editorInput.scrollTop;
    editorBackdrop.scrollLeft = editorInput.scrollLeft;
}

function updateCounts() {
    const text = editorInput.value;
    charCount.textContent = `${text.length} characters`;
    
    const words = text.trim().split(/\s+/).filter(w => w.length > 0);
    wordCount.textContent = `${words.length} words`;
}

function updateBackdrop() {
    const typedText = editorInput.value;
    let htmlContent = escapeHtml(typedText);
    
    if (activeSuggestion) {
        // Determine if we need to insert a space before the suggestion
        const needsSpace = typedText.length > 0 && !typedText.endsWith(' ') && !typedText.endsWith('\n');
        const spacePrefix = needsSpace ? ' ' : '';
        htmlContent += `<span class="ghost-suggestion">${spacePrefix}${escapeHtml(activeSuggestion)}</span>`;
    }
    
    // Add extra newline to force scroll alignment if ending with newline
    if (typedText.endsWith('\n')) {
        htmlContent += '\n';
    }
    
    editorBackdrop.innerHTML = htmlContent;
    syncScroll();
}

// -----------------------------------------------------------------------------
// API INTEGRATION
// -----------------------------------------------------------------------------
async function fetchPrediction() {
    const text = editorInput.value;
    if (!text.trim()) {
        clearSuggestions();
        return;
    }
    
    const startTime = performance.now();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                temperature: parseFloat(tempSlider.value),
                top_k: parseInt(kSlider.value),
                use_cache: cacheToggle.checked
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }
        
        const data = await response.json();
        const duration = Math.round(performance.now() - startTime);
        
        // Update latency metrics
        latencyVal.textContent = `${duration}ms`;
        if (data.from_cache) {
            cacheHits++;
            cacheHitsVal.textContent = cacheHits;
        }
        
        // Update state
        currentSuggestions = data.suggestions;
        if (currentSuggestions.length > 0) {
            activeSuggestion = currentSuggestions[0];
            renderSuggestionsList(data.suggestions, data.confidences);
        } else {
            clearSuggestions();
        }
        
        updateBackdrop();
        
    } catch (err) {
        console.error("Failed to fetch predictions:", err);
        latencyVal.textContent = "Error";
    }
}

function renderSuggestionsList(suggestions, confidences) {
    noSuggestions.classList.add('hidden');
    suggestionsList.classList.remove('hidden');
    
    suggestionsList.innerHTML = "";
    suggestions.forEach((word, idx) => {
        const conf = confidences[idx] || 0;
        
        const item = document.createElement('div');
        item.className = `suggestion-bar-item ${idx === 0 ? 'primary-suggestion' : ''}`;
        item.style.cursor = 'pointer';
        item.title = "Click to insert";
        
        // Let user click on any suggestion in the list to accept it
        item.addEventListener('mousedown', (e) => {
            e.preventDefault(); // Prevent text area blur
            acceptSpecificSuggestion(word);
        });
        
        item.innerHTML = `
            <div class="suggestion-bar-info">
                <span class="suggestion-word">${escapeHtml(word)}</span>
                <span class="suggestion-conf">${conf}%</span>
            </div>
            <div class="bar-outer">
                <div class="bar-inner" style="width: 0%"></div>
            </div>
        `;
        
        suggestionsList.appendChild(item);
        
        // Micro-animation for progress bar width fill
        setTimeout(() => {
            const bar = item.querySelector('.bar-inner');
            if (bar) bar.style.width = `${conf}%`;
        }, 50);
    });
}

function clearSuggestions() {
    activeSuggestion = "";
    currentSuggestions = [];
    noSuggestions.classList.remove('hidden');
    suggestionsList.classList.add('hidden');
    suggestionsList.innerHTML = "";
    updateBackdrop();
}

function acceptSpecificSuggestion(suggestion) {
    if (!suggestion) return;
    
    const text = editorInput.value;
    const needsSpace = text.length > 0 && !text.endsWith(' ') && !text.endsWith('\n');
    const spacePrefix = needsSpace ? ' ' : '';
    
    // Append selection
    editorInput.value = text + spacePrefix + suggestion + ' ';
    
    // Clear and prepare for next
    clearSuggestions();
    updateCounts();
    
    // Auto-focus back to input
    editorInput.focus();
}

// -----------------------------------------------------------------------------
// EVENT LISTENERS
// -----------------------------------------------------------------------------

// Monitor inputs
editorInput.addEventListener('input', () => {
    updateCounts();
    
    // Clear active suggestion immediately when typing starts
    activeSuggestion = "";
    updateBackdrop();
    
    // Reset debounce timer
    clearTimeout(debounceTimer);
    
    const text = editorInput.value;
    if (text.length === 0) {
        clearSuggestions();
        return;
    }
    
    // Fire api request after typing pauses
    debounceTimer = setTimeout(() => {
        fetchPrediction();
    }, DEBOUNCE_DELAY_MS);
});

// Capture special keyboard keys (Tab, Esc)
editorInput.addEventListener('keydown', (e) => {
    // Tab to accept suggestion
    if (e.key === 'Tab') {
        if (activeSuggestion) {
            e.preventDefault();
            acceptSpecificSuggestion(activeSuggestion);
        }
    }
    // Esc to discard suggestions
    else if (e.key === 'Escape') {
        e.preventDefault();
        clearSuggestions();
    }
});

// Scroll sync
editorInput.addEventListener('scroll', syncScroll);
editorInput.addEventListener('resize', syncScroll);

// Window resize sync
window.addEventListener('resize', syncScroll);

// Sliders and Configuration Changes
tempSlider.addEventListener('input', () => {
    tempVal.textContent = tempSlider.value;
});
kSlider.addEventListener('input', () => {
    kVal.textContent = kSlider.value;
});

// Initialization
updateCounts();
updateBackdrop();
syncScroll();
