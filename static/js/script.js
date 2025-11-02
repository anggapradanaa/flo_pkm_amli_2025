// Global variables
let updateInterval;
let mapUpdateInterval;
let draggedElement = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupDragAndDrop();
});

function initializeApp() {
    console.log('Initializing traffic monitor...');
    updateFrames();
    updateTrafficSummary();
    showAllLocations();
    
    updateInterval = setInterval(updateFrames, 2000);
    mapUpdateInterval = setInterval(updateTrafficSummary, 5000);
}

// Drag and Drop for CCTV cards
function setupDragAndDrop() {
    const grid = document.getElementById('cctv-grid');
    
    grid.addEventListener('dragover', (e) => {
        e.preventDefault();
        const afterElement = getDragAfterElement(grid, e.clientY);
        const dragging = document.querySelector('.dragging');
        
        if (afterElement == null) {
            grid.appendChild(dragging);
        } else {
            grid.insertBefore(dragging, afterElement);
        }
    });
}

function getDragAfterElement(container, y) {
    const draggableElements = [...container.querySelectorAll('.cctv-card:not(.dragging)')];
    
    return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        
        if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
        } else {
            return closest;
        }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}

function updateFrames() {
    fetch('/get_frames')
        .then(response => response.json())
        .then(data => {
            displayVideoFeeds(data);
        })
        .catch(error => {
            console.error('Error updating frames:', error);
        });
}

function displayVideoFeeds(data) {
    const grid = document.getElementById('cctv-grid');
    grid.innerHTML = '';

    Object.keys(data).forEach(nodeId => {
        const frame = data[nodeId];
        const card = document.createElement('div');
        card.className = 'cctv-card';
        card.draggable = true;
        
        card.addEventListener('dragstart', () => {
            card.classList.add('dragging');
        });
        
        card.addEventListener('dragend', () => {
            card.classList.remove('dragging');
        });
        
        card.innerHTML = `
            <img src="data:image/jpeg;base64,${frame.image}" 
                 alt="CCTV ${nodeId}" 
                 class="cctv-image">
            <div class="cctv-info">
                <div class="cctv-title">${getLocationName(nodeId)}</div>
                <div class="cctv-stats">
                    <span class="status-badge ${frame.status.toLowerCase()}">${frame.status}</span>
                    <span class="vehicle-count">
                        <i class="fas fa-car"></i>
                        ${frame.vehicle_count}
                    </span>
                </div>
            </div>
        `;
        
        grid.appendChild(card);
    });
}

function getLocationName(nodeId) {
    // NODE_LABELS is defined in the HTML template from Flask
    return NODE_LABELS[nodeId] || `Lokasi ${nodeId}`;
}

function updateTrafficSummary() {
    fetch('/get_traffic_summary')
        .then(response => response.json())
        .then(data => {
            animateNumber('total-vehicles', data.total_vehicles || 0);
            animateNumber('lancar-count', data.lancar_count || 0);
            animateNumber('padat-count', data.padat_count || 0);
            animateNumber('macet-count', data.macet_count || 0);
        })
        .catch(error => {
            console.error('Error updating summary:', error);
        });
}

function animateNumber(elementId, targetValue) {
    const element = document.getElementById(elementId);
    const currentValue = parseInt(element.textContent) || 0;
    
    if (currentValue === targetValue) return;
    
    const duration = 500;
    const steps = 20;
    const stepValue = (targetValue - currentValue) / steps;
    let currentStep = 0;
    
    const timer = setInterval(() => {
        currentStep++;
        const newValue = Math.round(currentValue + (stepValue * currentStep));
        element.textContent = newValue;
        
        if (currentStep >= steps) {
            element.textContent = targetValue;
            clearInterval(timer);
        }
    }, duration / steps);
}

function calculateRoute() {
    const startNode = parseInt(document.getElementById('start-location').value);
    const endNode = parseInt(document.getElementById('end-location').value);
    
    if (startNode === endNode) {
        alert('⚠️ Pilih lokasi yang berbeda!');
        return;
    }

    const btn = document.getElementById('calculate-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Menghitung...';

    document.getElementById('map-container').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Menghitung rute optimal dengan A*...</p>
        </div>
    `;
    
    document.getElementById('route-info').classList.remove('show');

    fetch('/get_route_map', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: startNode, end: endNode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayRouteMap(data);
            displayRouteInfo(data);
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('map-container').innerHTML = `
            <div class="loading">
                <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: var(--danger);"></i>
                <p>Error: ${error.message}</p>
            </div>
        `;
    })
    .finally(() => {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-route"></i> Hitung Rute Optimal';
    });
}

function displayRouteMap(data) {
    document.getElementById('map-container').innerHTML = data.map_html;
}

function displayRouteInfo(data) {
    const routeInfo = document.getElementById('route-info');
    const summary = document.getElementById('route-summary');
    const details = document.getElementById('route-details');

    summary.innerHTML = `
        <div class="route-metric">
            <i class="fas fa-route"></i>
            <div>
                <strong>Rute:</strong> ${data.route_nodes.join(' → ')}
            </div>
        </div>
        <div class="route-metric">
            <i class="fas fa-clock"></i>
            <div>
                <strong>Waktu:</strong> ${data.estimated_time} menit
            </div>
        </div>
        <div class="route-metric">
            <i class="fas fa-car"></i>
            <div>
                <strong>Total Kendaraan:</strong> ${data.total_vehicles}
            </div>
        </div>
    `;

    let detailsHTML = '<h4 style="margin-bottom: 0.5rem;">Detail Segmen:</h4>';
    data.route_details.forEach((segment, idx) => {
        detailsHTML += `
            <div class="route-step">
                <div>
                    <strong>${idx + 1}. ${segment.from} → ${segment.to}</strong>
                </div>
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <span class="status-badge ${segment.status}">${segment.status}</span>
                    <span style="color: #6B7280;">
                        <i class="fas fa-car"></i> ${segment.vehicle_count}
                    </span>
                </div>
            </div>
        `;
    });

    details.innerHTML = detailsHTML;
    routeInfo.classList.add('show');
}

function showAllLocations() {
    document.getElementById('map-container').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Memuat peta...</p>
        </div>
    `;
    
    document.getElementById('route-info').classList.remove('show');

    fetch('/get_interactive_map')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('map-container').innerHTML = data.map_html;
            } else {
                throw new Error(data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Cleanup
window.addEventListener('beforeunload', () => {
    if (updateInterval) clearInterval(updateInterval);
    if (mapUpdateInterval) clearInterval(mapUpdateInterval);
});

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        if (updateInterval) clearInterval(updateInterval);
        if (mapUpdateInterval) clearInterval(mapUpdateInterval);
    } else {
        updateInterval = setInterval(updateFrames, 2000);
        mapUpdateInterval = setInterval(updateTrafficSummary, 5000);
    }
});