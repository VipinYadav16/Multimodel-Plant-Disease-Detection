// script.js

// DOM Elements
const navLinks = document.querySelector('.nav-links');
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const resultsContainer = document.getElementById('resultsContainer');

// Scroll Animation
function animateOnScroll() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    elements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        const elementBottom = element.getBoundingClientRect().bottom;
        
        if (elementTop < window.innerHeight && elementBottom > 0) {
            element.classList.add('visible');
        }
    });
}

window.addEventListener('scroll', animateOnScroll);

// Image Upload and Analysis Functionality
if (uploadBox) {
    // Drag and Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => {
            uploadBox.classList.add('highlight');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => {
            uploadBox.classList.remove('highlight');
        });
    });

    uploadBox.addEventListener('drop', handleDrop);
    uploadBox.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (validateFile(file)) {
            displayPreview(file);
        }
    }
}

function validateFile(file) {
    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG or PNG)');
        return false;
    }

    // Check file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
        alert('Please upload an image smaller than 5MB');
        return false;
    }

    return true;
}

function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        uploadBox.style.display = 'none';
        previewContainer.style.display = 'block';
        resultsContainer.style.display = 'none';
    }
    
    reader.readAsDataURL(file);
}

// Analysis Simulation
if (analyzeBtn) {
    analyzeBtn.addEventListener('click', simulateAnalysis);
}

if (resetBtn) {
    resetBtn.addEventListener('click', resetUpload);
}

function simulateAnalysis() {
    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    // Simulate API call delay
    setTimeout(() => {
        // Mock results
        const results = {
            plantName: 'Tomato Plant (Solanum lycopersicum)',
            diseaseName: 'Early Blight',
            confidence: 94.5,
        };
        
        displayResults(results);
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Image';
    }, 2000);
}

function displayResults(results) {
    document.getElementById('plantName').textContent = results.plantName;
    document.getElementById('diseaseName').textContent = results.diseaseName;
    document.getElementById('confidenceScore').textContent = results.confidence;
    
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

function resetUpload() {
    uploadBox.style.display = 'block';
    previewContainer.style.display = 'none';
    resultsContainer.style.display = 'none';
    fileInput.value = '';
}

// Navbar Scroll Effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > lastScroll) {
        navbar.style.transform = 'translateY(-100%)';
    } else {
        navbar.style.transform = 'translateY(0)';
    }
    
    lastScroll = currentScroll;
});

// Add smooth scrolling to all links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Initialize animations on load
window.addEventListener('load', () => {
    document.body.classList.add('loaded');
    animateOnScroll();
});

// Add loading animation to all images
document.querySelectorAll('img').forEach(img => {
    img.addEventListener('load', function() {
        this.classList.add('loaded');
    });
});

// Add intersection observer for lazy loading and animations
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
};

const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

document.querySelectorAll('.animate-on-scroll').forEach(element => {
    observer.observe(element);
});


