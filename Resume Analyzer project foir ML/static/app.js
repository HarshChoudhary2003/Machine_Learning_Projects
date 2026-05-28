document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const uploadForm = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');
    
    // Drag and Drop Logic
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }, false);

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type === "application/pdf") {
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                fileInput.files = files; // Update input files if dropped
            } else {
                alert("Please upload a PDF file.");
            }
        }
    }

    // Form Submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert("Please select a resume PDF first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        
        const jd = document.getElementById('jdInput').value;
        if (jd) formData.append("job_description", jd);
        
        const skills = document.getElementById('skillsInput').value;
        if (skills) formData.append("skills", skills);

        // UI State
        resultsSection.classList.add('hidden');
        loader.classList.remove('hidden');

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error("Server error during analysis.");
            }
            
            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            alert("Error: " + error.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    // Render Results
    function displayResults(data) {
        resultsSection.classList.remove('hidden');
        
        // 1. Animate Score Circle
        animateScore(data.similarity_score);
        
        // 2. Render Entities
        const entityList = document.getElementById('entityList');
        entityList.innerHTML = '';
        for (const [key, value] of Object.entries(data.entities)) {
            let valStr = Array.isArray(value) ? (value.length > 0 ? value.join(', ') : 'None') : value;
            
            const li = document.createElement('li');
            li.className = 'entity-item';
            li.innerHTML = `
                <span class="entity-label">${key}</span>
                <span class="entity-value">${valStr}</span>
            `;
            entityList.appendChild(li);
        }
        
        // 3. Render Skills
        const foundContainer = document.getElementById('foundSkills');
        const missingContainer = document.getElementById('missingSkills');
        
        foundContainer.innerHTML = '';
        missingContainer.innerHTML = '';
        
        if (data.skills_found.length === 0) foundContainer.innerHTML = '<span style="color: #cbd5e1">None</span>';
        data.skills_found.forEach(skill => {
            const chip = document.createElement('div');
            chip.className = 'chip found';
            chip.textContent = skill;
            foundContainer.appendChild(chip);
        });
        
        if (data.skills_missing.length === 0) missingContainer.innerHTML = '<span style="color: #cbd5e1">None</span>';
        data.skills_missing.forEach(skill => {
            const chip = document.createElement('div');
            chip.className = 'chip missing';
            chip.textContent = skill;
            missingContainer.appendChild(chip);
        });
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function animateScore(targetScore) {
        let progressCircle = document.getElementById("progressCircle");
        let scoreValue = document.getElementById("scoreValue");
        
        let progressStartValue = 0;
        let progressEndValue = Math.round(targetScore);
        let speed = 20;
        
        if(progressEndValue === 0) {
            scoreValue.textContent = "0%";
            progressCircle.style.background = `conic-gradient(#8b5cf6 0deg, rgba(255, 255, 255, 0.1) 0deg)`;
            return;
        }
        
        let progress = setInterval(() => {
            progressStartValue++;
            
            scoreValue.textContent = `${progressStartValue}%`;
            progressCircle.style.background = `conic-gradient(
                #8b5cf6 ${progressStartValue * 3.6}deg,
                rgba(255, 255, 255, 0.1) ${progressStartValue * 3.6}deg
            )`;
            
            if (progressStartValue == progressEndValue) {
                clearInterval(progress);
            }
        }, speed);
    }
});
