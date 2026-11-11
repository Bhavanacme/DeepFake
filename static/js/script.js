/**
 * Deepfake Detector - Frontend Interactivity & Animations Script
 * Handles drag-and-drop, button glows, animated processing, file previews,
 * scroll-based fade-in animations, and counter animations.
 */

document.addEventListener('DOMContentLoaded', () => {

  // =========================================================================
  // 1. BUTTON CLICK & GLOW INTERACTION
  // =========================================================================
  const interactiveButtons = document.querySelectorAll(
    '.btn-primary-blue, .btn-outline-blue, .btn-detect-main, .btn-glow-primary'
  );

  interactiveButtons.forEach(btn => {
    btn.addEventListener('pointerdown', () => {
      btn.classList.add('btn-active-glow');
    });
    ['pointerup', 'pointercancel', 'mouseleave'].forEach(evt => {
      btn.addEventListener(evt, () => {
        btn.classList.remove('btn-active-glow');
      });
    });
  });

  // =========================================================================
  // 2. PASSWORD VISIBILITY TOGGLES
  // =========================================================================
  document.querySelectorAll('.password-toggle-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = btn.getAttribute('data-target');
      const input = document.getElementById(targetId);
      if (input) {
        if (input.type === 'password') {
          input.type = 'text';
          btn.innerHTML = '<i class="bi bi-eye-slash-fill text-primary"></i>';
        } else {
          input.type = 'password';
          btn.innerHTML = '<i class="bi bi-eye-fill"></i>';
        }
      }
    });
  });

  // =========================================================================
  // 3. DRAG & DROP FILE UPLOAD HANDLER
  // =========================================================================
  document.querySelectorAll('.upload-dropzone-box').forEach(dropzone => {
    const parentContainer = dropzone.closest('form') || document;
    const fileInput = parentContainer.querySelector('input[type="file"]');
    const fileInfoCard = parentContainer.querySelector('.selected-file-info');
    const fileNameDisplay = parentContainer.querySelector('.selected-file-name');
    const fileSizeDisplay = parentContainer.querySelector('.selected-file-size');
    const fileTypeBadge = parentContainer.querySelector('.selected-file-type-badge');
    const previewThumbnail = parentContainer.querySelector('.preview-thumbnail-img');
    const submitBtn = parentContainer.querySelector('.btn-detect-main, button[type="submit"]');
    const removeBtn = parentContainer.querySelector('.btn-remove-selected-file');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', (e) => {
      if (e.target !== fileInput) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.add('drag-over');
      }, false);
    });

    ['dragleave', 'drop'].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.remove('drag-over');
      }, false);
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        fileInput.files = files;
        updateSelectedFileInfo(files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        updateSelectedFileInfo(fileInput.files[0]);
      }
    });

    if (removeBtn) {
      removeBtn.addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        fileInput.value = '';
        if (fileInfoCard) fileInfoCard.classList.add('d-none');
        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.classList.add('opacity-50');
        }
      });
    }

    function updateSelectedFileInfo(file) {
      if (!file) return;
      const validExtensions = ['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'];
      const ext = file.name.split('.').pop().toLowerCase();

      if (!validExtensions.includes(ext)) {
        alert(`Unsupported format (.${ext}). Please upload JPG, PNG, MP4, AVI, or MOV.`);
        fileInput.value = '';
        if (fileInfoCard) fileInfoCard.classList.add('d-none');
        if (submitBtn) submitBtn.disabled = true;
        return;
      }

      const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
      if (file.size > 50 * 1024 * 1024) {
        alert(`File size (${sizeMB} MB) exceeds the 50 MB limit.`);
        fileInput.value = '';
        if (fileInfoCard) fileInfoCard.classList.add('d-none');
        if (submitBtn) submitBtn.disabled = true;
        return;
      }

      if (fileNameDisplay) fileNameDisplay.textContent = file.name;
      if (fileSizeDisplay) fileSizeDisplay.textContent = `${sizeMB} MB`;

      const isVideo = ['mp4', 'avi', 'mov', 'mkv'].includes(ext);
      if (fileTypeBadge) {
        fileTypeBadge.textContent = isVideo ? 'VIDEO' : 'IMAGE';
        fileTypeBadge.className = isVideo
          ? 'badge text-white px-2 py-1'
          : 'badge bg-primary px-2 py-1';
        if (isVideo) fileTypeBadge.style.backgroundColor = '#7c3aed';
        else fileTypeBadge.style.backgroundColor = '';
      }

      if (previewThumbnail) {
        if (!isVideo && file.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = (e) => {
            previewThumbnail.src = e.target.result;
            previewThumbnail.classList.remove('d-none');
          };
          reader.readAsDataURL(file);
        } else {
          previewThumbnail.classList.add('d-none');
        }
      }

      if (fileInfoCard) fileInfoCard.classList.remove('d-none');
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-50');
      }
    }
  });

  // =========================================================================
  // 4. DETECT BUTTON SUBMISSION & 5-STAGE ANIMATED PROCESSING OVERLAY
  // =========================================================================
  let isSubmitting = false;
  const processingOverlay = document.getElementById('processingOverlay');

  document.querySelectorAll('form[action*="detect"], #uploadForm').forEach(form => {
    form.addEventListener('submit', (e) => {
      if (isSubmitting) { e.preventDefault(); return; }

      const fileInput = form.querySelector('input[type="file"]');
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert('Please choose an image or video file before clicking Detect Deepfake.');
        return;
      }

      isSubmitting = true;
      const submitBtn = form.querySelector('.btn-detect-main, button[type="submit"]');
      if (submitBtn) {
        submitBtn.classList.add('is-analyzing');
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
          <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
          Analyzing Media...
        `;
      }

      if (processingOverlay) {
        processingOverlay.style.display = 'flex';
        const steps = [
          'step-upload', 'step-preprocess', 'step-face', 'step-ai', 'step-result'
        ].map(id => document.getElementById(id));

        const delays = [0, 600, 1300, 2200, 3100];
        steps.forEach((step, idx) => {
          if (!step) return;
          setTimeout(() => {
            // Mark previous as done
            if (idx > 0 && steps[idx - 1]) {
              steps[idx - 1].classList.remove('active');
              steps[idx - 1].classList.add('completed');
              steps[idx - 1].querySelector('.step-status').innerHTML =
                '<i class="bi bi-check-circle-fill text-success fs-5"></i>';
            }
            step.classList.add('active');
            step.querySelector('.step-status').innerHTML =
              '<span class="spinner-border spinner-border-sm text-primary"></span>';
          }, delays[idx]);
        });
      }
    });
  });

  // =========================================================================
  // 5. ANIMATE CONFIDENCE BARS ON RESULT LOAD
  // =========================================================================
  document.querySelectorAll('.animate-progress, .confidence-fill-bar').forEach(bar => {
    const targetWidth = bar.getAttribute('data-width');
    if (targetWidth) {
      setTimeout(() => { bar.style.width = targetWidth; }, 200);
    }
  });

  // =========================================================================
  // 6. AUTO-DISMISS ALERT BANNERS (6 seconds)
  // =========================================================================
  document.querySelectorAll('.alert-dismissible').forEach(alertEl => {
    setTimeout(() => {
      if (typeof bootstrap !== 'undefined' && bootstrap.Alert) {
        const bsAlert = bootstrap.Alert.getOrCreateInstance(alertEl);
        if (bsAlert) bsAlert.close();
      }
    }, 6000);
  });

  // =========================================================================
  // 7. SCROLL-BASED FADE-IN ANIMATIONS (IntersectionObserver)
  //    Cards and sections animate in as user scrolls down the page.
  // =========================================================================
  if (window.matchMedia('(prefers-reduced-motion: no-preference)').matches) {
    const observerOptions = {
      root: null,
      rootMargin: '0px 0px -50px 0px',
      threshold: 0.08
    };

    const fadeObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('fade-in-visible');
          fadeObserver.unobserve(entry.target);
        }
      });
    }, observerOptions);

    // Observe all cards and KPI sections
    document.querySelectorAll(
      '.custom-card, .kpi-card, .pipeline-step-card, .auth-card, ' +
      '.result-column-card, .upload-main-card, section > .container > .row > div'
    ).forEach(el => {
      if (!el.classList.contains('fade-in-visible')) {
        el.style.opacity = '0';
        el.style.transform = 'translateY(18px)';
        el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        fadeObserver.observe(el);
      }
    });
  }

  // =========================================================================
  // 8. ANIMATED KPI COUNTER NUMBERS (Dashboard)
  // =========================================================================
  function animateCounter(el, target, duration = 900) {
    const start = 0;
    const startTime = performance.now();
    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      el.textContent = Math.round(start + (target - start) * eased);
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

  document.querySelectorAll('.kpi-card h3, .kpi-card h4').forEach(el => {
    const val = parseInt(el.textContent.trim(), 10);
    if (!isNaN(val) && val > 0) {
      el.textContent = '0';
      animateCounter(el, val, 900);
    }
  });

});

// =========================================================================
// CSS helper: fade-in-visible class applied by IntersectionObserver
// =========================================================================
(function () {
  const style = document.createElement('style');
  style.textContent = '.fade-in-visible { opacity: 1 !important; transform: translateY(0) !important; }';
  document.head.appendChild(style);
})();
