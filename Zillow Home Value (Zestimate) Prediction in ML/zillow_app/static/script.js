document.addEventListener('DOMContentLoaded', async () => {
    // Elements
    const formGrid = document.getElementById('dynamic-form');
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const submitBtnText = submitBtn.querySelector('span');

    // State Elements
    const initialState = document.getElementById('initial-state');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');
    const resultOutput = document.getElementById('prediction-output');

    // Helper: Show/Hide States
    const setViewState = (state) => {
        initialState.style.display = 'none';
        loadingState.style.display = 'none';
        resultState.style.display = 'none';

        if (state === 'initial') {
            initialState.style.display = 'block';
            initialState.style.animation = 'fadeIn 0.5s ease forwards';
        }
        if (state === 'loading') {
            loadingState.style.display = 'block';
            loadingState.style.animation = 'fadeIn 0.5s ease forwards';
        }
        if (state === 'result') {
            resultState.style.display = 'block';
            resultState.style.animation = 'fadeIn 0.5s ease forwards';
        }
    };

    // Helper: Animate Number
    const animateValue = (obj, start, end, duration) => {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(4);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    };

    // 1. Load Features Metadata
    try {
        const response = await fetch('/api/meta');
        const data = await response.json();

        if (data.error) {
            formGrid.innerHTML = `
                <div style="grid-column: 1/-1; padding: 2rem; text-align: center; color: var(--error);">
                    <i class="ri-error-warning-line" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                    <p>Failed to load model metadata: ${data.error}</p>
                </div>`;
            return;
        }

        // Clear Loading Spinner
        formGrid.innerHTML = '';

        // Build Form Fields
        data.schema.forEach((field, index) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'input-wrapper';
            wrapper.style.animation = `fadeIn 0.5s ease forwards ${index * 0.05}s`;
            wrapper.style.opacity = '0'; // Start hidden for animation

            const label = document.createElement('label');
            label.htmlFor = field.name;
            label.textContent = field.label; // Backend provides formatted label

            let input;

            if (field.type === 'select') {
                input = document.createElement('select');
                input.className = 'input-field';
                input.id = field.name;
                input.name = field.name;

                // Add default placeholder option
                const defaultOpt = document.createElement('option');
                defaultOpt.value = "";
                defaultOpt.disabled = true;
                defaultOpt.selected = true;
                defaultOpt.textContent = "Select...";
                input.appendChild(defaultOpt);

                field.options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt;
                    input.appendChild(option);
                });
            } else {
                input = document.createElement('input');
                input.className = 'input-field';
                input.type = 'number';
                input.id = field.name;
                input.name = field.name;
                input.step = 'any';
                input.placeholder = "0.00";

                if (field.default !== undefined) input.value = field.default.toFixed(2);
                if (field.min !== undefined) input.min = field.min;
                if (field.max !== undefined) input.max = field.max;
            }

            wrapper.appendChild(label);
            wrapper.appendChild(input);
            formGrid.appendChild(wrapper);
        });

        // Enable Submit Button
        submitBtn.disabled = false;

        // Add fadeIn animation style
        const styleSheet = document.createElement("style");
        styleSheet.innerText = `
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        `;
        document.head.appendChild(styleSheet);


    } catch (err) {
        console.error("Feature load error:", err);
        formGrid.innerHTML = `
            <div style="grid-column: 1/-1; padding: 2rem; text-align: center; color: var(--error);">
                <i class="ri-wifi-off-line" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                <p>Could not connect to the backend server.</p>
            </div>`;
    }

    // 2. Handle Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Updates
        setViewState('loading');
        submitBtn.disabled = true;
        submitBtnText.textContent = "Calculating...";

        // Collect Data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // Convert to number if it looks like one, otherwise keep string
            // The backend handles conversion too, but good to be safe
            if (!isNaN(value) && value.trim() !== '') {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        });

        try {
            // Artificial delay for better UX (so user sees 'analyzing')
            await new Promise(r => setTimeout(r, 800));

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.error) {
                alert(`Prediction Error: ${result.error}`);
                setViewState('initial'); // Go back
            } else {
                setViewState('result');
                const predValue = parseFloat(result.prediction);
                animateValue(resultOutput, 0, predValue, 1000);
            }

        } catch (err) {
            console.error(err);
            alert('An unexpected error occurred. Please try again.');
            setViewState('initial');
        } finally {
            submitBtn.disabled = false;
            submitBtnText.textContent = "Generate Prediction";
        }
    });
});
