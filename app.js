// Application Data
const applicationData = {
  trafficTypes: [
    {type: "Normal", color: "#4CAF50", baseRate: 85},
    {type: "DDoS", color: "#F44336", baseRate: 5},
    {type: "Malware", color: "#FF9800", baseRate: 3},
    {type: "Reconnaissance", color: "#9C27B0", baseRate: 4},
    {type: "Data Exfiltration", color: "#E91E63", baseRate: 3}
  ],
  firewallRules: [
    {id: 1, rule: "Block IP range 192.168.1.0/24", priority: "High", confidence: 0.95, action: "BLOCK"},
    {id: 2, rule: "Allow HTTP traffic on port 80", priority: "Medium", confidence: 0.88, action: "ALLOW"},
    {id: 3, rule: "Rate limit ICMP packets", priority: "Medium", confidence: 0.92, action: "LIMIT"},
    {id: 4, rule: "Block suspicious User-Agent patterns", priority: "High", confidence: 0.89, action: "BLOCK"}
  ],
  performanceMetrics: {
    accuracy: 98.7,
    falsePositiveRate: 0.13,
    responseTime: 12,
    threatsStopped: 1247,
    totalTraffic: 156789
  },
  rlAgentStats: {
    trainingEpisodes: 50000,
    currentReward: 0.847,
    learningRate: 0.001,
    explorationRate: 0.1,
    memorySize: 10000
  },
  systemComponents: [
    {name: "LSTM-CNN Model", status: "Active", load: 67},
    {name: "DQN Agent", status: "Learning", load: 54},
    {name: "SDN Controller", status: "Connected", load: 32},
    {name: "Feature Extractor", status: "Processing", load: 78}
  ]
};

// Global state
let simulationActive = false;
let charts = {};
let animationFrames = [];
let currentTrafficIntensity = 5;
let autoUpdateInterval = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
  initializeNavigation();
  initializeCharts();
  initializeDashboard();
  initializeSimulation();
  initializeRulesManagement();
  initializeArchitecture();
  startRealTimeUpdates();
});

// Navigation handling
function initializeNavigation() {
  const navButtons = document.querySelectorAll('.nav-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  navButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.getAttribute('data-tab');
      
      // Update active nav button
      navButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
      
      // Update active tab content
      tabContents.forEach(tab => tab.classList.remove('active'));
      document.getElementById(targetTab).classList.add('active');
      
      // Refresh charts when switching tabs
      setTimeout(() => {
        Object.values(charts).forEach(chart => {
          if (chart && chart.resize) {
            chart.resize();
          }
        });
      }, 100);
    });
  });
}

// Initialize charts
function initializeCharts() {
  // Traffic Distribution Chart
  const trafficCtx = document.getElementById('trafficChart');
  if (trafficCtx) {
    charts.traffic = new Chart(trafficCtx, {
      type: 'doughnut',
      data: {
        labels: applicationData.trafficTypes.map(t => t.type),
        datasets: [{
          data: applicationData.trafficTypes.map(t => t.baseRate),
          backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom'
          }
        }
      }
    });
  }

  // Training Progress Chart
  const trainingCtx = document.getElementById('trainingChart');
  if (trainingCtx) {
    const episodes = Array.from({length: 50}, (_, i) => i * 1000);
    const rewards = episodes.map(ep => Math.min(0.9, 0.1 + (ep / 50000) * 0.8 + Math.random() * 0.1));
    
    charts.training = new Chart(trainingCtx, {
      type: 'line',
      data: {
        labels: episodes,
        datasets: [{
          label: 'Average Reward',
          data: rewards,
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 1
          }
        }
      }
    });
  }

  // Rule Effectiveness Chart
  const effectivenessCtx = document.getElementById('effectivenessChart');
  if (effectivenessCtx) {
    charts.effectiveness = new Chart(effectivenessCtx, {
      type: 'bar',
      data: {
        labels: applicationData.firewallRules.map(r => `Rule ${r.id}`),
        datasets: [{
          label: 'Effectiveness %',
          data: applicationData.firewallRules.map(r => r.confidence * 100),
          backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100
          }
        }
      }
    });
  }

  // Accuracy Over Time Chart
  const accuracyCtx = document.getElementById('accuracyChart');
  if (accuracyCtx) {
    const timeLabels = Array.from({length: 24}, (_, i) => `${i}:00`);
    const rlAccuracy = timeLabels.map(() => 95 + Math.random() * 4);
    const staticAccuracy = timeLabels.map(() => 85 + Math.random() * 3);

    charts.accuracy = new Chart(accuracyCtx, {
      type: 'line',
      data: {
        labels: timeLabels,
        datasets: [{
          label: 'RL Firewall',
          data: rlAccuracy,
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.1)',
          tension: 0.4
        }, {
          label: 'Static Firewall',
          data: staticAccuracy,

          borderColor: '#B4413C',
          backgroundColor: 'rgba(180, 65, 60, 0.1)',
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 80,
            max: 100
          }
        }
      }
    });
  }

  // Threat Distribution Chart
  const threatCtx = document.getElementById('threatChart');
  if (threatCtx) {
    charts.threat = new Chart(threatCtx, {
      type: 'pie',
      data: {
        labels: ['DDoS', 'Malware', 'Reconnaissance', 'Data Exfiltration', 'Other'],
        datasets: [{
          data: [35, 25, 20, 15, 5],
          backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom'
          }
        }
      }
    });
  }

  // Comparison Chart
  const comparisonCtx = document.getElementById('comparisonChart');
  if (comparisonCtx) {
    charts.comparison = new Chart(comparisonCtx, {
      type: 'radar',
      data: {
        labels: ['Accuracy', 'Speed', 'Adaptability', 'False Positives', 'Threat Detection'],
        datasets: [{
          label: 'RL Firewall',
          data: [95, 85, 98, 95, 92],
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.2)'
        }, {
          label: 'Static Firewall',
          data: [78, 95, 20, 70, 65],
          borderColor: '#B4413C',
          backgroundColor: 'rgba(180, 65, 60, 0.2)'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            max: 100
          }
        }
      }
    });
  }

  // Response Time Chart
  const responseCtx = document.getElementById('responseChart');
  if (responseCtx) {
    const timeData = Array.from({length: 20}, (_, i) => ({
      x: i * 5,
      y: 8 + Math.random() * 8
    }));

    charts.response = new Chart(responseCtx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Response Time (ms)',
          data: timeData,
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            title: {
              display: true,
              text: 'Time (minutes)'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Response Time (ms)'
            }
          }
        }
      }
    });
  }
}

// Initialize dashboard
function initializeDashboard() {
  updateSystemTime();
  renderActiveRules();
  startTrafficVisualization();
}

// Update system time
function updateSystemTime() {
  const timeElement = document.getElementById('systemTime');
  if (timeElement) {
    const now = new Date();
    timeElement.textContent = now.toLocaleTimeString();
  }
}

// Render active rules
function renderActiveRules() {
  const container = document.getElementById('activeRulesList');
  if (container) {
    container.innerHTML = applicationData.firewallRules.slice(0, 3).map(rule => `
      <div class="rule-item">
        <span class="rule-text">${rule.rule}</span>
        <span class="rule-priority ${rule.priority.toLowerCase()}">${rule.priority}</span>
      </div>
    `).join('');
  }
}

// Traffic visualization
function startTrafficVisualization() {
  const container = document.getElementById('trafficFlow');
  if (!container) return;

  function createParticle() {
    const particle = document.createElement('div');
    const types = ['normal', 'threat', 'blocked'];
    const weights = [0.85, 0.1, 0.05];
    
    let random = Math.random();
    let type = 'normal';
    let cumulative = 0;
    
    for (let i = 0; i < types.length; i++) {
      cumulative += weights[i];
      if (random <= cumulative) {
        type = types[i];
        break;
      }
    }
    
    particle.className = `traffic-particle ${type}`;
    particle.style.top = Math.random() * 140 + 'px';
    particle.style.animationDuration = (2 + Math.random() * 2) + 's';
    
    container.appendChild(particle);
    
    setTimeout(() => {
      if (particle.parentNode) {
        particle.parentNode.removeChild(particle);
      }
    }, 4000);
  }

  // Create particles periodically
  const particleInterval = setInterval(() => {
    if (Math.random() < currentTrafficIntensity / 10) {
      createParticle();
    }
  }, 200);

  animationFrames.push(particleInterval);
}

// Initialize simulation controls
function initializeSimulation() {
  const startBtn = document.getElementById('startSimulation');
  const stopBtn = document.getElementById('stopSimulation');
  const intensitySlider = document.getElementById('intensitySlider');
  const intensityValue = document.getElementById('intensityValue');
  const clearLogBtn = document.getElementById('clearLog');

  if (startBtn) {
    startBtn.addEventListener('click', startSimulation);
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', stopSimulation);
  }

  if (intensitySlider) {
    intensitySlider.addEventListener('input', (e) => {
      currentTrafficIntensity = parseInt(e.target.value);
      if (intensityValue) {
        intensityValue.textContent = e.target.value;
      }
    });
  }

  if (clearLogBtn) {
    clearLogBtn.addEventListener('click', () => {
      const logContainer = document.getElementById('simulationLog');
      if (logContainer) {
        logContainer.innerHTML = '';
      }
    });
  }
}

// Simulation functions
function startSimulation() {
  simulationActive = true;
  document.getElementById('startSimulation').disabled = true;
  document.getElementById('stopSimulation').disabled = false;
  
  addLogEntry('Simulation started', 'info');
  
  // Simulate traffic events
  const simulationInterval = setInterval(() => {
    if (!simulationActive) {
      clearInterval(simulationInterval);
      return;
    }
    
    const trafficType = document.getElementById('trafficType').value;
    simulateTrafficEvent(trafficType);
  }, 1000 + Math.random() * 2000);

  animationFrames.push(simulationInterval);
}

function stopSimulation() {
  simulationActive = false;
  document.getElementById('startSimulation').disabled = false;
  document.getElementById('stopSimulation').disabled = true;
  
  addLogEntry('Simulation stopped', 'info');
}

function simulateTrafficEvent(type) {
  const events = {
    normal: ['HTTP request processed', 'HTTPS connection established', 'DNS query resolved'],
    ddos: ['DDoS attack detected from 192.168.1.100', 'High volume traffic blocked', 'Rate limiting applied'],
    malware: ['Malicious payload detected', 'Suspicious file transfer blocked', 'C&C communication intercepted'],
    recon: ['Port scan detected', 'Reconnaissance attempt blocked', 'Suspicious probe activity'],
    exfiltration: ['Data exfiltration attempt', 'Unauthorized data transfer blocked', 'Sensitive data access denied']
  };
  
  const typeEvents = events[type] || events.normal;
  const event = typeEvents[Math.floor(Math.random() * typeEvents.length)];
  const eventType = type === 'normal' ? 'info' : 'threat';
  
  addLogEntry(event, eventType);
  
  // Update metrics
  updateMetrics();
}

function addLogEntry(message, type) {
  const logContainer = document.getElementById('simulationLog');
  if (!logContainer) return;

  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  
  const timestamp = new Date().toLocaleTimeString();
  entry.innerHTML = `
    <span class="log-timestamp">[${timestamp}]</span>
    <span class="log-message">${message}</span>
  `;
  
  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;
  
  // Limit log entries
  const entries = logContainer.querySelectorAll('.log-entry');
  if (entries.length > 50) {
    entries[0].remove();
  }
}

// Update metrics with random variations
function updateMetrics() {
  const metrics = applicationData.performanceMetrics;
  
  // Small random variations
  metrics.accuracy += (Math.random() - 0.5) * 0.1;
  metrics.accuracy = Math.max(95, Math.min(99.5, metrics.accuracy));
  
  metrics.responseTime += (Math.random() - 0.5) * 2;
  metrics.responseTime = Math.max(8, Math.min(20, metrics.responseTime));
  
  metrics.falsePositiveRate += (Math.random() - 0.5) * 0.02;
  metrics.falsePositiveRate = Math.max(0.05, Math.min(0.3, metrics.falsePositiveRate));
  
  // Update display
  updateDashboardMetrics();
}

function updateDashboardMetrics() {
  const elements = {
    accuracy: document.getElementById('accuracy'),
    responseTime: document.getElementById('responseTime'),
    fpRate: document.getElementById('fpRate'),
    throughput: document.getElementById('throughput'),
    threatsDetected: document.getElementById('threatsDetected'),
    threatsBlocked: document.getElementById('threatsBlocked'),
    falsePositives: document.getElementById('falsePositives')
  };
  
  const metrics = applicationData.performanceMetrics;
  
  if (elements.accuracy) {
    elements.accuracy.textContent = metrics.accuracy.toFixed(1) + '%';
  }
  if (elements.responseTime) {
    elements.responseTime.textContent = Math.round(metrics.responseTime) + 'ms';
  }
  if (elements.fpRate) {
    elements.fpRate.textContent = metrics.falsePositiveRate.toFixed(2) + '%';
  }
  if (elements.throughput) {
    const throughputValue = (Math.random() * 50 + 100).toFixed(0);
    elements.throughput.textContent = throughputValue + "";
  }
  
  // Update threat counters with small increments
  const currentThreats = parseInt(elements.threatsDetected?.textContent || '0');
  const currentBlocked = parseInt(elements.threatsBlocked?.textContent || '0');
  const currentFP = parseInt(elements.falsePositives?.textContent || '0');
  
  if (Math.random() < 0.3) { // 30% chance to increment
    if (elements.threatsDetected) {
      elements.threatsDetected.textContent = currentThreats + 1;
    }
    if (elements.threatsBlocked && Math.random() < 0.97) { // 97% block rate
      elements.threatsBlocked.textContent = currentBlocked + 1;
    } else if (elements.falsePositives && Math.random() < 0.20) {
      elements.falsePositives.textContent = currentFP + 1;
    }
  }
}

// Rules management
function initializeRulesManagement() {
  renderRulesTable();

  const addRuleBtn = document.getElementById('addRule');
  if (addRuleBtn) {
    addRuleBtn.addEventListener('click', () => {
      addNewRule();
    });
  }

  const manualUpdateBtn = document.getElementById('manualUpdateBtn');
  if (manualUpdateBtn) {
    manualUpdateBtn.addEventListener('click', () => {
      manualUpdate();
    });
  }

  const autoUpdateSwitch = document.getElementById('autoUpdateSwitch');
  if (autoUpdateSwitch) {
    autoUpdateSwitch.addEventListener('change', (e) => {
      if (e.target.checked) {
        startAutomaticUpdates();
      } else {
        stopAutomaticUpdates();
      }
    });

    if (autoUpdateSwitch.checked) {
      startAutomaticUpdates();
    }
  }
}

function renderRulesTable() {
  const tbody = document.getElementById('rulesTableBody');
  if (!tbody) return;

  tbody.innerHTML = applicationData.firewallRules.map(rule => `
    <div class="table-row">
      <div class="rule-text">${rule.rule}</div>
      <div class="rule-priority ${rule.priority.toLowerCase()}">${rule.priority}</div>
      <div class="rule-confidence">${(rule.confidence * 100).toFixed(1)}%</div>
      <div class="rule-action ${rule.action.toLowerCase()}">${rule.action}</div>
      <div class="rule-controls">
        <button class="btn btn--sm btn--outline" onclick="editRule(${rule.id})">Edit</button>
        <button class="btn btn--sm btn--outline" onclick="deleteRule(${rule.id})">Delete</button>
      </div>
    </div>
  `).join('');
}

function addNewRule() {
  const newRule = {
    id: Math.max(...applicationData.firewallRules.map(r => r.id)) + 1,
    rule: "Block suspicious traffic pattern #" + Math.floor(Math.random() * 1000),
    priority: ['High', 'Medium', 'Low'][Math.floor(Math.random() * 3)],
    confidence: 0.7 + Math.random() * 0.3,
    action: ['BLOCK', 'ALLOW', 'LIMIT'][Math.floor(Math.random() * 3)]
  };
  
  applicationData.firewallRules.push(newRule);
  renderRulesTable();
  addRuleHistoryEntry(`Added rule: ${newRule.rule}`);
}

function editRule(id) {
  const rule = applicationData.firewallRules.find(r => r.id === id);
  if (rule) {
    rule.confidence = 0.7 + Math.random() * 0.3;
    renderRulesTable();
    addRuleHistoryEntry(`Modified rule ${id}: confidence updated to ${(rule.confidence * 100).toFixed(1)}%`);
  }
}

function deleteRule(id) {
  const ruleIndex = applicationData.firewallRules.findIndex(r => r.id === id);
  if (ruleIndex > -1) {
    const rule = applicationData.firewallRules[ruleIndex];
    applicationData.firewallRules.splice(ruleIndex, 1);
    renderRulesTable();
    addRuleHistoryEntry(`Deleted rule: ${rule.rule}`);
  }
}

function addRuleHistoryEntry(message) {
  const historyContainer = document.getElementById('ruleHistory');
  if (!historyContainer) return;

  const entry = document.createElement('div');
  entry.className = 'history-item';
  
  const timestamp = new Date().toLocaleTimeString();
  entry.innerHTML = `
    <div class="history-timestamp">[${timestamp}]</div>
    <div>${message}</div>
  `;
  
  historyContainer.insertBefore(entry, historyContainer.firstChild);
  
  // Limit history entries
  const entries = historyContainer.querySelectorAll('.history-item');
  if (entries.length > 20) {
    entries[entries.length - 1].remove();
  }
}

function manualUpdate() {
  addLogEntry('Manual rule update triggered', 'info');
  automaticRuleUpdate(); // Reuse the same logic for adding a rule
}

function startAutomaticUpdates() {
  if (autoUpdateInterval) {
    clearInterval(autoUpdateInterval);
  }
  autoUpdateInterval = setInterval(automaticRuleUpdate, 50000); // every 50 seconds
  addRuleHistoryEntry('Automatic rule updates enabled.');
}

function stopAutomaticUpdates() {
  if (autoUpdateInterval) {
    clearInterval(autoUpdateInterval);
    autoUpdateInterval = null;
  }
  addRuleHistoryEntry('Automatic rule updates disabled.');
}

function automaticRuleUpdate() {
  const newRule = {
    id: Math.max(...applicationData.firewallRules.map(r => r.id)) + 1,
    rule: "Auto-discovered pattern: block UDP flood on port " + (5000 + Math.floor(Math.random() * 1000)),
    priority: ['High', 'Medium'][Math.floor(Math.random() * 2)],
    confidence: 0.85 + Math.random() * 0.15,
    action: 'BLOCK'
  };

  applicationData.firewallRules.push(newRule);
  renderRulesTable();
  addRuleHistoryEntry(`New rule automatically added: ${newRule.rule}`);
}

// Initialize architecture view
function initializeArchitecture() {
  renderComponentStatus();
}

function renderComponentStatus() {
  const container = document.getElementById('componentsList');
  if (!container) return;

  container.innerHTML = applicationData.systemComponents.map(component => `
    <div class="component-item">
      <div class="component-name">${component.name}</div>
      <div class="component-status">
        <div class="component-indicator ${component.status.toLowerCase()}"></div>
        <span>${component.status}</span>
        <span class="component-load">${component.load}%</span>
      </div>
    </div>
  `).join('');
}

// Real-time updates
function startRealTimeUpdates() {
  // Update time every second
  setInterval(updateSystemTime, 1000);
  
  // Update metrics every 5 seconds
  setInterval(updateMetrics, 5000);
  
  // Update component loads every 10 seconds
  setInterval(() => {
    applicationData.systemComponents.forEach(component => {
      component.load += (Math.random() - 0.5) * 10;
      component.load = Math.max(10, Math.min(95, component.load));
    });
    renderComponentStatus();
  }, 10000);
  
  // Update RL agent stats every 15 seconds
  setInterval(() => {
    const stats = applicationData.rlAgentStats;
    stats.currentReward += (Math.random() - 0.5) * 0.1;
    stats.currentReward = Math.max(0.5, Math.min(1.0, stats.currentReward));
    
    const rewardElement = document.getElementById('currentReward');
    if (rewardElement) {
      rewardElement.textContent = '+' + Math.round(stats.currentReward * 1000);
    }
    
    // Update policy confidence bars
    const confidenceBars = document.querySelectorAll('.confidence-bar');
    confidenceBars.forEach(bar => {
      const newWidth = 70 + Math.random() * 25;
      bar.style.width = newWidth + '%';
      const valueSpan = bar.nextElementSibling;
      if (valueSpan) {
        valueSpan.textContent = Math.round(newWidth) + '%';
      }
    });
  }, 15000);
  
  // Update traffic rate display
  setInterval(() => {
    const rateElement = document.getElementById('trafficRate');
    if (rateElement) {
      const baseRate = 1200;
      const variation = Math.floor(Math.random() * 200 - 100);
      const newRate = baseRate + variation;
      rateElement.textContent = newRate.toLocaleString() + ' pps';
    }
  }, 3000);
}

// Cleanup function
window.addEventListener('beforeunload', () => {
  animationFrames.forEach(frame => {
    if (typeof frame === 'number') {
      clearInterval(frame);
    }
  });
});