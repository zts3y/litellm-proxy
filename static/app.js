document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const modelSelect = document.getElementById('model-select');
    const systemPrompt = document.getElementById('system-prompt');
    const streamToggle = document.getElementById('stream-toggle');
    const tempSlider = document.getElementById('temperature-slider');
    const tempValue = document.getElementById('temperature-value');
    const maxTokensInput = document.getElementById('max-tokens-input');
    const gcpStatus = document.getElementById('gcp-status');
    const statusIndicator = document.querySelector('.status-indicator');
    const activeModelTitle = document.getElementById('active-model-title');
    const activeModelDesc = document.getElementById('active-model-desc');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const chatMessages = document.getElementById('chat-messages');
    const emptyState = document.getElementById('empty-state');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const quickPromptBtns = document.querySelectorAll('.quick-prompt-btn');

    // Conversation State
    let messages = [];
    let isGenerating = false;

    // Auto-adjust textarea height as user types
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
        
        // Enable/disable send button based on input length
        sendBtn.disabled = !chatInput.value.trim() || isGenerating || !modelSelect.value;
    });

    // Handle CTRL/CMD + Enter key to submit
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            // Check if mac command key or windows control key is pressed
            if (e.metaKey || e.ctrlKey) {
                e.preventDefault();
                if (chatInput.value.trim() && !isGenerating && modelSelect.value) {
                    submitMessage();
                }
            }
        }
    });

    // Update temperature label on slider move
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });

    // Initialize: Fetch available models
    async function init() {
        try {
            gcpStatus.textContent = 'Connecting...';
            const response = await fetch('/api/models');
            if (!response.ok) throw new Error('Failed to load models');
            
            const models = await response.json();
            
            // Populate select dropdown
            modelSelect.innerHTML = '';
            if (models.length === 0) {
                modelSelect.innerHTML = '<option value="" disabled>No models available</option>';
                return;
            }

            models.forEach(model => {
                const opt = document.createElement('option');
                opt.value = model.id;
                opt.textContent = model.name;
                // Default to gemini-2.5-flash if present, otherwise first model
                if (model.id === 'vertex_ai/gemini-2.5-flash') {
                    opt.selected = true;
                }
                modelSelect.appendChild(opt);
            });

            // Update model display header
            updateModelHeader();

            // Set connected status
            statusIndicator.classList.add('connected');
            gcpStatus.textContent = 'GCP Vertex AI Connected';
            
            // Enable chat inputs
            chatInput.disabled = false;
            sendBtn.disabled = !chatInput.value.trim();

        } catch (error) {
            console.error('Error during initial setup:', error);
            gcpStatus.textContent = 'Connection Error';
            statusIndicator.classList.remove('connected');
            statusIndicator.style.color = 'var(--color-error)';
            
            // Create fallback option in select
            modelSelect.innerHTML = `
                <option value="vertex_ai/gemini-2.5-flash" selected>Gemini 2.5 Flash (Fallback)</option>
                <option value="vertex_ai/gemini-2.5-pro">Gemini 2.5 Pro (Fallback)</option>
                <option value="vertex_ai/gemini-3.5-flash">Gemini 3.5 Flash (Fallback)</option>
                <option value="vertex_ai/gemini-3-flash-preview">Gemini 3 Flash Preview (Fallback)</option>
                <option value="vertex_ai/gemini-3-pro-preview">Gemini 3 Pro Preview (Fallback)</option>
            `;
            updateModelHeader();
            chatInput.disabled = false;
        }
    }

    // Change model update event
    modelSelect.addEventListener('change', () => {
        updateModelHeader();
    });

    function updateModelHeader() {
        const selectedOpt = modelSelect.options[modelSelect.selectedIndex];
        if (selectedOpt) {
            activeModelTitle.textContent = selectedOpt.textContent;
            activeModelDesc.textContent = `Running endpoint: ${selectedOpt.value}`;
        }
    }

    // Clear Chat Handler
    clearChatBtn.addEventListener('click', () => {
        messages = [];
        // Remove all message rows, keep only the empty state if it's there, or recreate it
        const welcomeState = chatMessages.querySelector('.empty-state');
        chatMessages.innerHTML = '';
        if (welcomeState) {
            chatMessages.appendChild(welcomeState);
        } else {
            chatMessages.appendChild(emptyState);
            emptyState.style.display = 'flex';
        }
        clearChatBtn.blur();
    });

    // Quick prompt buttons
    quickPromptBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.textContent;
            chatInput.dispatchEvent(new Event('input'));
            chatInput.focus();
        });
    });

    // Formatting simple markdown-like syntax
    function formatMarkdown(text) {
        if (!text) return '';
        
        // Escape HTML to prevent XSS
        let escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
            
        // Fenced code blocks
        escaped = escaped.replace(/```([a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const displayLang = lang ? `<span class="code-lang">${lang}</span>` : '';
            return `<pre>${displayLang}<code>${code.trim()}</code></pre>`;
        });
        
        // Inline code
        escaped = escaped.replace(/`([^`\n]+)`/g, '<code>$1</code>');
        
        // Simple bold / italic
        escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
        // Split paragraphs by double newlines, wrap in <p>
        // Use double newline, or fallback single newlines for block splits
        const paragraphs = escaped.split(/\n\n+/);
        return paragraphs.map(p => {
            if (p.trim().startsWith('<pre>')) return p; // Don't wrap code blocks in paragraphs
            return `<p>${p.trim().replace(/\n/g, '<br>')}</p>`;
        }).join('');
    }

    // Scroll to bottom helper
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Message Element Creation Helper
    function appendMessage(role, content = '') {
        // Hide welcome state on first message
        if (emptyState.style.display !== 'none') {
            emptyState.style.display = 'none';
        }

        const row = document.createElement('div');
        row.className = `message-row ${role === 'user' ? 'user' : 'bot'}`;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        messageDiv.innerHTML = formatMarkdown(content);
        
        row.appendChild(messageDiv);
        chatMessages.appendChild(row);
        scrollToBottom();
        return messageDiv;
    }

    // Append Typing Indicator Helper
    function appendTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'message-row bot temp-typing-indicator';
        
        const indicator = document.createElement('div');
        indicator.className = 'message typing-indicator';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        
        row.appendChild(indicator);
        chatMessages.appendChild(row);
        scrollToBottom();
        return row;
    }

    // Submit message handling
    async function submitMessage() {
        const text = chatInput.value.trim();
        if (!text || isGenerating) return;

        isGenerating = true;
        
        // Add User message to state and UI
        messages.push({ role: 'user', content: text });
        appendMessage('user', text);
        
        // Clear input and reset height
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Show typing indicator
        const typingIndicatorRow = appendTypingIndicator();

        const model = modelSelect.value;
        const stream = streamToggle.checked;
        const system = systemPrompt.value.trim() || null;
        const temperature = parseFloat(tempSlider.value);
        const maxTokens = parseInt(maxTokensInput.value) || null;

        const payload = {
            model,
            messages,
            stream,
            system_prompt: system,
            temperature,
            max_tokens: maxTokens
        };

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            // Remove typing indicator once response starts or fails
            typingIndicatorRow.remove();

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'API Call failed');
            }

            if (stream) {
                // Streaming flow (Server-Sent Events)
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";
                let botMessageContent = "";
                let botMessageElement = appendMessage('bot', '');

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    
                    // Split by double newline as SSE events are separated by \n\n
                    let boundary = buffer.indexOf('\n\n');
                    while (boundary !== -1) {
                        const event = buffer.substring(0, boundary).trim();
                        buffer = buffer.substring(boundary + 2);
                        boundary = buffer.indexOf('\n\n');

                        if (event.startsWith('data: ')) {
                            const dataStr = event.slice(6).trim();
                            if (dataStr === '[DONE]') {
                                break;
                            }
                            try {
                                const parsed = JSON.parse(dataStr);
                                if (parsed.error) {
                                    throw new Error(parsed.error);
                                }
                                if (parsed.content) {
                                    botMessageContent += parsed.content;
                                    botMessageElement.innerHTML = formatMarkdown(botMessageContent);
                                    scrollToBottom();
                                }
                            } catch (e) {
                                // JSON parsing might fail on partial outputs, skip
                                console.warn('JSON parse error in SSE stream', e);
                            }
                        }
                    }
                }
                
                // Add final response to memory history
                if (botMessageContent) {
                    messages.push({ role: 'assistant', content: botMessageContent });
                }

            } else {
                // Standard JSON response
                const result = await response.json();
                const botContent = result.content || '';
                appendMessage('bot', botContent);
                messages.push({ role: 'assistant', content: botContent });
            }

        } catch (err) {
            console.error('Error during message completion:', err);
            // Re-remove indicator in case of early error
            if (typingIndicatorRow.parentNode) {
                typingIndicatorRow.remove();
            }
            appendMessage('bot', `⚠️ **Error:** ${err.message || 'Could not communicate with Vertex AI endpoint.'}`);
        } finally {
            isGenerating = false;
            sendBtn.disabled = !chatInput.value.trim();
        }
    }

    // Submit form action
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitMessage();
    });

    // Run setup
    init();
});
