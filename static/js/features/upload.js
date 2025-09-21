// static/js/features/upload.js
(() => {
  const form = document.getElementById('upload-form');
  const input = document.getElementById('upload-input');
  const img   = document.getElementById('preview-image');
  const ph    = document.getElementById('stage-placeholder');
  const loadBtn = document.getElementById('load-into-editor');
  const uploadBtn = document.getElementById('upload-btn');

  // Ensure the button doesn't submit the form accidentally
  if (loadBtn) loadBtn.setAttribute('type', 'button');

  async function loadIntoEditor(file) {
    if (!file) return;

    // 1) Local preview
    const url = URL.createObjectURL(file);
    img.src = url;

    // 2) Wait for decode
    try { 
      await img.decode(); 
    } catch (_) {
      await new Promise(res => img.complete ? res() : img.addEventListener('load', res, { once: true }));
    }

    // 3) Reset any CSS transforms from previous images
    img.style.transform = '';
    img.style.transformOrigin = '';

    // 4) Update state
    APP_STATE.hasImage = true;
    APP_STATE.edits = {
      exposure:0, contrast:0, whites:0, blacks:0,
      temperature:0, vibrance:0, saturation:0,
      sharpness:0, grain:0, dehaze:0
    };
    APP_STATE.crop = { zoom:1, x:0, y:0, rot:0 };

    // 5) Update UI - IMPORTANT: Add has-image to both body and stage
    document.body.classList.add('has-image');
    const stage = document.getElementById('stage');
    if (stage) stage.classList.add('has-image');

    // Hide placeholder
    ph?.setAttribute('aria-hidden', 'true');
    if (ph) ph.style.display = 'none';

    // NO MORE before/after button logic

    // Switch to edit tab
    const editTab = document.querySelector('.tab[data-tab="edit"]');
    if (editTab) editTab.click();

    // 6) Upload to server
    try {
      const fd = new FormData();
      fd.append('image', file);
      const res = await fetch('/api/upload', { method: 'POST', body: fd });
      const result = await res.json();
      if (!res.ok) {
        alert(result.error || 'Upload failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
    }
  }

  // If your UX uses a button
  loadBtn?.addEventListener('click', async (e) => {
    e.preventDefault();
    const f = input.files?.[0];
    await loadIntoEditor(f);
  });

  // If your UX uploads via <form>, keep this too
  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = input.files?.[0];
    await loadIntoEditor(f);
  });

  // Button opens file picker
  uploadBtn?.addEventListener('click', () => {
    input.click();
  });

  // Auto-load when file is selected
  input?.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await loadIntoEditor(file);
      // Clear input for next selection
      input.value = '';
    }
  });
})();
