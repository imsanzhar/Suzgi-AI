document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const btn = form.querySelector('button');
    const originalBtnText = btn.innerHTML;
    
    btn.innerHTML = 'Өңделуде...';
    btn.style.opacity = '0.8';
    btn.disabled = true;

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Сервер қатесі');
        }
        
        const resultSection = document.getElementById('result-section');
        const probEl = document.getElementById('res-prob');
        const badgeEl = document.getElementById('res-badge');
        const nnEl = document.getElementById('res-nn');
        const xgbEl = document.getElementById('res-xgb');
        const progressBar = document.getElementById('progress-bar');
        
        const probValue = result.probability * 100;
        
        probEl.textContent = `${probValue.toFixed(2)}%`;
        
        if (nnEl && xgbEl) {
            nnEl.textContent = `${(result.nn * 100).toFixed(1)}%`;
            xgbEl.textContent = `${(result.xgb * 100).toFixed(1)}%`;
        }

        badgeEl.style.color = 'white';
        
        if (result.risk === "Жоғары қауіп" || probValue >= 30) {
            badgeEl.textContent = '🚨 ЖОҒАРЫ ҚАУІП';
            badgeEl.style.backgroundColor = 'var(--danger)';
            progressBar.style.backgroundColor = 'var(--danger)';
        } else {
            badgeEl.textContent = '✅ ТӨМЕН ҚАУІП';
            badgeEl.style.backgroundColor = 'var(--success)';
            progressBar.style.backgroundColor = 'var(--success)';
        }

        resultSection.style.display = 'block';
        
        setTimeout(() => {
            progressBar.style.width = `${probValue}%`;
        }, 100);

        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        alert(`Қате орын алды: ${error.message}`);
    } finally {
        btn.innerHTML = originalBtnText;
        btn.style.opacity = '1';
        btn.disabled = false;
    }
});