// SciRAG Chat Interface JavaScript

const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
let isWaitingForResponse = false;
let retrievedSources = []; // Store sources from latest query
let conversationHistory = []; // Store conversation history for multi-turn

// Generate a unique session ID (fallback for browsers without crypto.randomUUID)
function generateSessionId() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    // Fallback: generate a random ID
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
let sessionId = generateSessionId();

// Configure marked.js for markdown rendering
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,  // Convert \n to <br>
        gfm: true,     // GitHub Flavored Markdown
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (e) {
                    console.error('Highlight error:', e);
                }
            }
            return code;
        }
    });
}

// Load collections and MCP status on page load
function initializePage() {
    console.log('initializePage called');
    loadCollections();
    loadMcpStatus().then(() => {
        console.log('loadMcpStatus completed successfully');
    }).catch(err => {
        console.error('Failed to load MCP status on page load:', err);
        const statusDiv = document.getElementById('mcpServerStatus');
        if (statusDiv) {
            statusDiv.innerHTML = '<span class="text-warning small"><i class="bi bi-exclamation-triangle"></i> Failed to load status</span>';
        }
    });
}

// Run initialization - handle case where DOMContentLoaded already fired
console.log('Script loaded, readyState:', document.readyState);
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePage);
} else {
    // DOM already loaded, run immediately
    initializePage();
}

// Load available collections into the dropdown
async function loadCollections() {
    const select = document.getElementById('collectionSelect');
    if (!select) return;

    try {
        const response = await fetch('/api/collections');
        const data = await response.json();

        // Clear existing options except "All Collections"
        select.innerHTML = '<option value="">All Collections</option>';

        if (data.collections && data.collections.length > 0) {
            data.collections.forEach(collection => {
                const option = document.createElement('option');
                option.value = collection;
                option.textContent = collection;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Failed to load collections:', error);
    }
}

// Refresh collections list
function refreshCollections() {
    loadCollections();
}

// Get selected collection (empty string means all collections)
function getSelectedCollection() {
    const select = document.getElementById('collectionSelect');
    return select ? select.value : '';
}

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

    // Get selected collection
    const collection = getSelectedCollection();
    
    // Add user message to conversation history
    conversationHistory.push({ role: 'user', content: message });
    
    try {
        const requestBody = {
            query: message,
            messages: conversationHistory,  // Send full conversation history
            session_id: sessionId,
            top_k: 5
        };
        
        // Only include collection if one is selected
        if (collection) {
            requestBody.collection = collection;
        }

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        // Remove loading indicator
        removeLoading(loadingId);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get response');
        }
        
        const data = await response.json();
        addMessage(data.response, 'assistant', data.sources);
        
        // Add assistant response to conversation history
        conversationHistory.push({ role: 'assistant', content: data.response });
        
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
        // Remove the failed user message from history
        conversationHistory.pop();
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
    
    // Render markdown for assistant messages, escape HTML for user messages
    let renderedContent;
    if (role === 'assistant' && typeof marked !== 'undefined') {
        renderedContent = marked.parse(content);
    } else {
        renderedContent = escapeHtml(content);
    }
    
    let messageHTML = `
        <div class="message-content${role === 'assistant' ? ' markdown-body' : ''}">
            ${renderedContent}
    `;
    
    if (sources && sources.length > 0) {
        messageHTML += '<div class="message-sources">';
        messageHTML += '<i class="bi bi-files"></i> <strong>Sources:</strong> ';
        sources.forEach(source => {
            const collection = source.collection || 'DocumentChunks';
            messageHTML += `<span class="source-tag">${escapeHtml(source.source)} <span class="collection-badge">${escapeHtml(collection)}</span></span>`;
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

// Clear chat history and reset conversation
function clearChat() {
    // Reset conversation history
    conversationHistory = [];
    sessionId = generateSessionId();
    retrievedSources = [];
    
    // Clear chat messages display
    chatMessages.innerHTML = `
        <div class="text-center py-5 welcome-message">
            <h3 class="text-primary mb-3"><i class="bi bi-robot"></i> Welcome to SciRAG!</h3>
            <p class="text-muted mb-4">
                Ask me anything about your ingested documents.<br>
                I'll search through the knowledge base and provide answers with sources.
            </p>
            <div class="d-grid gap-2 col-md-8 mx-auto">
                <button class="btn btn-outline-primary text-start" onclick="sendExampleQuestion(this)">
                    What are the main findings?
                </button>
                <button class="btn btn-outline-primary text-start" onclick="sendExampleQuestion(this)">
                    Explain the methodology used
                </button>
                <button class="btn btn-outline-primary text-start" onclick="sendExampleQuestion(this)">
                    What are the key conclusions?
                </button>
            </div>
        </div>
    `;
    
    // Clear sources tab
    const sourcesContent = document.getElementById('sourcesContent');
    if (sourcesContent) {
        sourcesContent.innerHTML = `
            <div class="text-center py-5 text-muted">
                <i class="bi bi-file-earmark-text" style="font-size: 3rem;"></i>
                <p class="mt-3">No sources yet. Ask a question to see retrieved documents.</p>
            </div>
        `;
    }
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
                        <th><i class="bi bi-folder2"></i> Collection</th>
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
        const collection = source.collection || 'DocumentChunks';
        tableHTML += `
            <tr>
                <td class="source-filename">${escapeHtml(source.source || 'Unknown')}</td>
                <td><span class="badge bg-secondary">${escapeHtml(collection)}</span></td>
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

// Load MCP server status
async function loadMcpStatus() {
    const statusDiv = document.getElementById('mcpServerStatus');
    if (!statusDiv) {
        console.warn('mcpServerStatus element not found');
        return;
    }

    try {
        const response = await fetch('/api/mcp-status');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();

        let html = '<div class="d-flex flex-wrap align-items-center gap-1">';

        // Connected servers
        if (data.connected && data.connected.length > 0) {
            data.connected.forEach((server, index) => {
                const toolCount = server.tools ? server.tools.length : 0;
                const popoverId = `mcp-popover-${index}`;
                
                // Build tools list HTML for popover content
                let toolsContent = '';
                if (server.tools && server.tools.length > 0) {
                    toolsContent = '<ul class=&quot;tool-list&quot;>';
                    server.tools.forEach(tool => {
                        toolsContent += `<li><span class=&quot;tool-name&quot;><i class=&quot;bi bi-gear-fill&quot;></i> ${escapeHtml(tool)}</span></li>`;
                    });
                    toolsContent += '</ul>';
                } else {
                    toolsContent = '<span class=&quot;text-muted&quot;>No tools available</span>';
                }
                
                html += `
                    <span class="badge bg-success mcp-server-badge" 
                          id="${popoverId}"
                          data-bs-toggle="popover" 
                          data-bs-trigger="hover focus"
                          data-bs-placement="top"
                          data-bs-html="true"
                          data-bs-title="<i class='bi bi-tools'></i> ${escapeHtml(server.name)} Tools"
                          data-bs-content="${toolsContent}">
                        <i class="bi bi-check-circle-fill"></i> ${escapeHtml(server.name)}
                        <span class="badge bg-light text-success ms-1">${toolCount} tools</span>
                    </span>
                `;
            });
        }

        // Failed servers - show in red to indicate they were expected but unavailable
        if (data.failed && data.failed.length > 0) {
            data.failed.forEach((server, index) => {
                const errorMessage = server.error || 'Connection failed';
                const popoverId = `mcp-failed-popover-${index}`;
                const serverName = server.name || server.url;
                const errorContent = `<div class=&quot;small&quot;><strong>Server:</strong> ${escapeHtml(serverName)}<br><strong>URL:</strong> ${escapeHtml(server.url)}<br><strong>Error:</strong> ${escapeHtml(errorMessage)}</div>`;
                
                html += `
                    <span class="badge bg-danger mcp-server-badge"
                          id="${popoverId}"
                          data-bs-toggle="popover"
                          data-bs-trigger="hover focus"
                          data-bs-placement="top"
                          data-bs-html="true"
                          data-bs-title="<i class='bi bi-exclamation-triangle-fill'></i> Server Unavailable"
                          data-bs-content="${errorContent}">
                        <i class="bi bi-x-circle-fill"></i> ${escapeHtml(serverName)}
                        <span class="badge bg-light text-danger ms-1">offline</span>
                    </span>
                `;
            });
        }

        html += '</div>';

        // No servers configured
        if ((!data.connected || data.connected.length === 0) && 
            (!data.failed || data.failed.length === 0)) {
            html = '<span class="text-muted small"><i class="bi bi-info-circle"></i> No MCP servers configured</span>';
        }

        statusDiv.innerHTML = html;
        
        // Initialize Bootstrap popovers for MCP server badges
        initMcpPopovers();
        
    } catch (error) {
        console.error('Failed to load MCP status:', error);
        statusDiv.innerHTML = '<span class="text-warning small"><i class="bi bi-exclamation-triangle"></i> Failed to check server status</span>';
    }
}

// Initialize Bootstrap popovers for MCP server badges
function initMcpPopovers() {
    const popoverElements = document.querySelectorAll('.mcp-server-badge[data-bs-toggle="popover"]');
    popoverElements.forEach(el => {
        new bootstrap.Popover(el, {
            container: 'body',
            customClass: 'mcp-tools-popover'
        });
    });
}

// Refresh MCP status
function refreshMcpStatus() {
    const statusDiv = document.getElementById('mcpServerStatus');
    if (statusDiv) {
        statusDiv.innerHTML = '<span class="text-muted small"><i class="bi bi-hourglass-split"></i> Checking...</span>';
    }
    loadMcpStatus();
}

// Focus input on load
messageInput.focus();
