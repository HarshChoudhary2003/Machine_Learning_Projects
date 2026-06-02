document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const resultTitle = document.getElementById('result-title');
    const probabilityBar = document.getElementById('probability-bar');
    const resultProbability = document.getElementById('result-probability');
    const resetBtn = document.getElementById('reset-btn');
    const submitBtn = document.getElementById('submit-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Change button state
        const originalBtnText = submitBtn.textContent;
        submitBtn.textContent = 'Predicting...';
        submitBtn.disabled = true;

        const data = {
            daily_time: document.getElementById('daily_time').value,
            age: document.getElementById('age').value,
            area_income: document.getElementById('area_income').value,
            daily_internet: document.getElementById('daily_internet').value,
            male: document.getElementById('male').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            // Hide form, show results
            form.classList.add('hidden');
            resultContainer.classList.remove('hidden');

            // Reset bar width for animation
            probabilityBar.style.width = '0%';
            
            // Set content based on prediction
            const percentage = (result.probability * 100).toFixed(1);
            
            setTimeout(() => {
                probabilityBar.style.width = `${percentage}%`;
                
                if (result.prediction === 1) {
                    resultTitle.textContent = "Will Click!";
                    resultTitle.style.color = "var(--success)";
                    probabilityBar.className = "probability-bar positive";
                } else {
                    resultTitle.textContent = "Won't Click";
                    resultTitle.style.color = "var(--error)";
                    probabilityBar.className = "probability-bar negative";
                }
                
                resultProbability.textContent = `Probability: ${percentage}%`;
            }, 50);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during prediction. Please try again.');
        } finally {
            submitBtn.textContent = originalBtnText;
            submitBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        form.reset();
        resultContainer.classList.add('hidden');
        form.classList.remove('hidden');
    });
});
