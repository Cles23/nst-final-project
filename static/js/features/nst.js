(() => {
  // Get all the UI elements we need
  const elements = {
    upload: document.getElementById('style-upload'),
    uploadBtn: document.getElementById('style-upload-btn'),
    preview: document.getElementById('style-preview'),
    previewImg: document.getElementById('style-preview-img'),
    startBtn: document.getElementById('start-nst-btn'),
    progress: document.getElementById('nst-progress'),
    method: document.getElementById('nst-method'),
    steps: document.getElementById('nst-steps'),
    stepsOut: document.getElementById('nst-steps-out'),
    styleStrength: document.getElementById('style-strength'),
    styleStrengthOut: document.getElementById('style-strength-out'),
    stepsControl: document.getElementById('steps-control'),
    styleStrengthControl: document.getElementById('style-strength-control'),
    methodHint: document.getElementById('method-hint'),
    processingTime: document.getElementById('processing-time')
  };

  let currentStylePath = null;
  let currentJobId = null;
  let startTime = null;

  // When user changes method, update the UI
  elements.method?.addEventListener('change', () => {
    const method = elements.method.value;
    updateUIForMethod(method);
    updateStartButton();
  });

  function updateUIForMethod(method) {
    if (method === 'gatys') {
      // Show iterations slider for Gatys
      elements.stepsControl.style.display = 'block';
      elements.styleStrengthControl.style.display = 'none';
      elements.methodHint.textContent = 'Gatys: 2-5 minutes, high quality but slow.';
    } else if (method === 'adain') {
      // Show strength slider for AdaIN
      elements.stepsControl.style.display = 'none';
      elements.styleStrengthControl.style.display = 'block';
      elements.methodHint.textContent = 'AdaIN: 30-60 seconds, good quality and fast.';
    }
  }

  // Update the numbers shown next to sliders
  elements.steps?.addEventListener('input', () => {
    elements.stepsOut.textContent = elements.steps.value;
  });

  elements.styleStrength?.addEventListener('input', () => {
    elements.styleStrengthOut.textContent = elements.styleStrength.value;
  });

  // Handle style image upload
  elements.uploadBtn?.addEventListener('click', () => elements.upload.click());
  
  elements.upload?.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Upload to server
    const formData = new FormData();
    formData.append('style', file);
    
    try {
      const res = await fetch('/api/nst/upload-style', { method: 'POST', body: formData });
      const result = await res.json();
      
      if (res.ok) {
        currentStylePath = result.path;
        // Show preview of uploaded style
        elements.previewImg.src = URL.createObjectURL(file);
        elements.preview.style.display = 'block';
        updateStartButton();
      }
    } catch (err) {
      alert('Style upload failed');
    }
  });

  function updateStartButton() {
    // Can only start if we have both content image and style image
    let canStart = APP_STATE.hasImage && currentStylePath;
    elements.startBtn.disabled = !canStart;
  }

  // Start the style transfer
  elements.startBtn?.addEventListener('click', async () => {
    const method = elements.method?.value || 'gatys';
    
    if (!APP_STATE.hasImage || !currentStylePath) return;

    setProcessing(true);
    startTime = Date.now();
    
    try {
      // Prepare data to send to server
      const params = {
        method: method,
        edits: APP_STATE.edits || {},
        crop: APP_STATE.crop || {}
      };
      
      // Add method-specific settings
      if (method === 'gatys') {
        params.style_path = currentStylePath;
        params.steps = parseInt(elements.steps?.value || 300);
      } else if (method === 'adain') {
        params.style_path = currentStylePath;
        params.strength = parseFloat(elements.styleStrength?.value || 1.0);
      }

      console.log(`Starting ${method.toUpperCase()} NST:`, params);

      // Start the job on server
      const res = await fetch('/api/nst/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
      });

      const result = await res.json();
      if (!res.ok) {
        throw new Error(result.error || 'NST failed to start');
      }

      currentJobId = result.job_id;
      pollStatus(); // Start checking if it's done
    } catch (err) {
      alert('NST failed: ' + err.message);
      setProcessing(false);
    }
  });

  // Keep checking if the job is done
  async function pollStatus() {
    if (!currentJobId) return;

    // Update timer
    if (startTime && elements.processingTime) {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      elements.processingTime.textContent = `(${elapsed}s)`;
    }

    try {
      const res = await fetch(`/api/nst/status/${currentJobId}`);
      const status = await res.json();

      if (status.status === 'complete') {
        // Job is done!
        const totalTime = Math.floor((Date.now() - startTime) / 1000);
        console.log(`${status.method?.toUpperCase()} completed in ${totalTime}s`);
        
        // Refresh the preview to show result
        if (window.EDIT?.schedulePreview) {
          window.EDIT.schedulePreview();
        }
        setProcessing(false);
        
      } else if (status.status === 'error') {
        alert(`NST failed: ${status.error || 'Unknown error'}`);
        setProcessing(false);
        
      } else {
        // Still running, check again in 1 second
        setTimeout(pollStatus, 1000);
      }

    } catch (err) {
      console.error('Status check failed:', err);
      setProcessing(false);
    }
  }

  function setProcessing(processing) {
    // Update UI based on whether we're processing or not
    elements.startBtn.disabled = processing;
    elements.startBtn.textContent = processing ? 'Processing...' : 'Apply Style Transfer';
    elements.progress.style.display = processing ? 'block' : 'none';
    
    if (!processing) {
      startTime = null;
      if (elements.processingTime) elements.processingTime.textContent = '';
      updateStartButton();
    }
  }

  // Set up initial state when page loads
  document.addEventListener('DOMContentLoaded', () => {
    const initialMethod = elements.method?.value || 'gatys';
    updateUIForMethod(initialMethod);
    updateStartButton();
  });

  // Update button when image is loaded
  document.addEventListener('image-loaded', () => {
    updateStartButton();
  });
})();