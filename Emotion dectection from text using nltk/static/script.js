document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const resultSection = document.getElementById('result-section');
    const emotionEmoji = document.getElementById('emotion-emoji');
    const emotionLabel = document.getElementById('emotion-label');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidencePercentage = document.getElementById('confidence-percentage');
    
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');

    // Emotion to Emoji and Color Theme Mapping
    const emotionMap = {
        'joy': { emoji: '✨', colorVar: '--theme-joy' },
        'sadness': { emoji: '🌧️', colorVar: '--theme-sadness' },
        'anger': { emoji: '🔥', colorVar: '--theme-anger' },
        'fear': { emoji: '👁️', colorVar: '--theme-fear' },
        'surprise': { emoji: '⚡', colorVar: '--theme-surprise' },
        'disgust': { emoji: '🤢', colorVar: '--theme-disgust' },
        'neutral': { emoji: '💨', colorVar: '--theme-neutral' }
    };

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        // Set loading state
        setLoading(true);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayResult(data.emotion, data.score);
        } catch (error) {
            console.error('Error analyzing text:', error);
            alert('An error occurred while analyzing the text. Check the console for details.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            
            // Reset result section if showing
            resultSection.classList.remove('show');
            setTimeout(() => {
                if (isLoading) resultSection.classList.add('hidden');
            }, 300); // Wait for transition
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    }

    function displayResult(emotion, score) {
        const emotionLower = emotion.toLowerCase();
        const mapping = emotionMap[emotionLower] || emotionMap['neutral'];
        
        // Update Theme
        document.documentElement.style.setProperty('--active-theme', `var(${mapping.colorVar})`);
        
        // Update UI Elements
        emotionEmoji.textContent = mapping.emoji;
        
        // Force reflow for pop animation restart
        emotionEmoji.style.animation = 'none';
        emotionEmoji.offsetHeight; /* trigger reflow */
        emotionEmoji.style.animation = null;

        emotionLabel.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        
        const percentage = Math.round(score * 100);
        confidencePercentage.textContent = `${percentage}%`;
        
        // Show result section
        resultSection.classList.remove('hidden');
        
        // Slight delay to allow display:block to apply before animating opacity/transform
        setTimeout(() => {
            resultSection.classList.add('show');
            // Animate confidence bar
            confidenceFill.style.width = '0%';
            setTimeout(() => {
                confidenceFill.style.width = `${percentage}%`;
            }, 100);
        }, 10);
    }
});
