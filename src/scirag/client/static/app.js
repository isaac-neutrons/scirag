// SciRAG Chat Interface JavaScript

const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
let isWaitingForResponse = false;
let retrievedSources = []; // Store sources from latest query

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function sendExampleQuestion(element) {
    messageInput.value = element.textContent.trim();
    sendMessage();
}

async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isWaitingForResponse) {
        return;
    }

    // Clear welcome message on first interaction
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Disable input while waiting
    isWaitingForResponse = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
    
    // Show loading indicator
    const loadingId = showLoading();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: message,
                top_k: 5
            })
        });
        
        // Remove loading indicator
        removeLoading(loadingId);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get response');
        }
        
        const data = await response.json();
        addMessage(data.response, 'assistant', data.sources);
        
        // Store sources for the Sources tab
        if (data.sources && data.sources.length > 0) {
            retrievedSources = data.sources;
            // Update sources tab if it's currently open
            if (document.getElementById('sourcesTab').classList.contains('active')) {
                displaySources();
            }
        }
        
    } catch (error) {
        removeLoading(loadingId);
        showError('Error: ' + error.message);
    } finally {
        // Re-enable input
        isWaitingForResponse = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
}

function addMessage(content, role, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    let messageHTML = `
        <div class="message-content">
            ${escapeHtml(content)}
    `;
    
    if (sources && sources.length > 0) {
        messageHTML += '<div class="message-sources">';
        messageHTML += '<i class="bi bi-files"></i> <strong>Sources:</strong> ';
        sources.forEach(source => {
            messageHTML += `<span class="source-tag">${escapeHtml(source.source)} (chunk ${source.chunk_index})</span>`;
        });
        messageHTML += '</div>';
    }
    
    messageHTML += '</div>';
    messageDiv.innerHTML = messageHTML;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    const loadingId = 'loading-' + Date.now();
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message assistant';
    loadingDiv.innerHTML = `
        <div class="loading-indicator">
            <div class="loading-dots">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return loadingId;
}

function removeLoading(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.innerHTML = `
        <i class="bi bi-exclamation-triangle-fill"></i> ${escapeHtml(message)}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Tab switching functionality
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    if (tabName === 'chat') {
        document.getElementById('chatTab').classList.add('active');
    } else if (tabName === 'sources') {
        document.getElementById('sourcesTab').classList.add('active');
        // Display sources from latest query
        displaySources();
    }
}

// Display sources from latest query in the Sources tab
function displaySources() {
    const sourcesContent = document.getElementById('sourcesContent');

    if (retrievedSources.length === 0) {
        sourcesContent.innerHTML = `
            <div class="text-center py-5">
                <i class="bi bi-inbox" style="font-size: 3rem; color: #6c757d;"></i>
                <h5 class="mt-3 text-muted">No Sources Yet</h5>
                <p class="text-muted">
                    Ask a question in the Chat tab to retrieve relevant sources.<br>
                    The retrieved sources will appear here.
                </p>
            </div>
        `;
        return;
    }

    // Build sources table using Bootstrap
    let tableHTML = `
        <div class="table-responsive">
            <table class="table table-hover sources-table">
                <thead>
                    <tr>
                        <th><i class="bi bi-file-text"></i> Source Document</th>
                        <th><i class="bi bi-hash"></i> Chunk</th>
                        <th><i class="bi bi-star-fill"></i> Score</th>
                        <th><i class="bi bi-eye"></i> Preview</th>
                    </tr>
                </thead>
                <tbody>
    `;

    retrievedSources.forEach(source => {
        const content = source.content || '';
        const preview = content.length > 100 ? content.substring(0, 100) + '...' : content;
        tableHTML += `
            <tr>
                <td class="source-filename">${escapeHtml(source.source || 'Unknown')}</td>
                <td>${source.chunk_index !== undefined ? source.chunk_index : 'N/A'}</td>
                <td>
                    ${source.score !== undefined ? source.score.toFixed(3) : 'N/A'}
                </td>
                <td class="content-preview" title="${escapeHtml(content)}">${escapeHtml(preview)}</td>
            </tr>
        `;
    });

    tableHTML += `
                </tbody>
            </table>
        </div>
    `;

    sourcesContent.innerHTML = tableHTML;
}

// Format file size in human-readable format
function formatFileSize(bytes) {
    if (!bytes) return 'N/A';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

// Format Unix timestamp to readable date
function formatDate(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// Focus input on load
messageInput.focus();
