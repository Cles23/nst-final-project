// Client-side rasterize (basic): applies filters/rotation, optional resize, then downloads
(() => {
  const dl = document.getElementById('download-btn');
  const fmtSel = document.getElementById('export-format');
  const img = document.getElementById('preview-image');

  // Use server-side export (always quality 100)
  dl?.addEventListener('click', async () => {
    if (!APP_STATE.hasImage) return;

    const fmt = (fmtSel?.value || 'png').toLowerCase();

    try {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          edits: APP_STATE.edits || {},
          crop: APP_STATE.crop || {},
          format: fmt,
          quality: 100
        })
      });

      if (!res.ok) throw new Error('Export failed');
      
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `edited.${fmt}`;
      a.click();
      URL.revokeObjectURL(blob);
    } catch (err) {
      alert('Export failed: ' + err.message);
    }
  });

  // "Start Over" button
  document.getElementById('restart-btn')?.addEventListener('click', () => {
    if (!confirm('Are you sure you want to start over? All changes will be lost.')) {
      return;
    }
    
    const ph = document.getElementById('stage-placeholder');
    const stage = document.getElementById('stage');
    const toggleBtn = document.getElementById('before-after-btn');
    
    img.src = '';
    APP_STATE.hasImage = false;
    APP_STATE.edits = {};
    APP_STATE.crop = { zoom:1, x:0, y:0, rot:0 };
    
    // Reset all slider values
    const allSliders = document.querySelectorAll('input[type="range"]');
    allSliders.forEach(slider => {
      const defaultVal = slider.id === 'zoom' ? 1 : 0;
      slider.value = defaultVal;
      const output = document.getElementById(`${slider.id}-out`);
      if (output) output.textContent = defaultVal;
    });
    
    ph.style.display = '';
    ph.setAttribute('aria-hidden', 'false');
    
    // Disable the app interface
    document.body.classList.remove('has-image');
    
    // Hide the floating toggle button
    if (stage) stage.classList.remove('has-image');
    if (toggleBtn) toggleBtn.style.display = 'none';
  });
})();
