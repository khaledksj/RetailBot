// Shop Manual Chatbot - Frontend Application

class ChatbotApp {
    constructor() {
        this.currentLanguage = 'en';
        this.currentTheme = 'light';
        this.websocket = null;
        this.selectedFiles = [];
        this.isUploading = false;
        this.isChatting = false;
        
        this.translations = {
            en: {
                upload_documents: "Upload Documents",
                chat: "Chat",
                upload_pdf_manuals: "Upload PDF Manuals",
                drag_drop_files: "Drag & Drop PDF files here",
                or_click_to_browse: "or click to browse",
                choose_files: "Choose Files",
                force_reingest: "Force re-ingest (overwrite existing documents)",
                upload_documents: "Upload Documents",
                manual_assistant: "Manual Assistant",
                temperature: "Temperature",
                type_your_question: "Type your question...",
                sources: "Sources",
                welcome_message: "Hello! I'm here to help you with questions about your uploaded manuals. Please upload some PDF documents first, then ask me anything!",
                processing: "Processing...",
                uploading: "Uploading files...",
                file_uploaded: "File uploaded successfully",
                file_failed: "Failed to upload file",
                file_skipped: "File skipped (already exists)",
                no_files_selected: "No files selected",
                upload_error: "Upload error",
                connection_error: "Connection error",
                sending_message: "Sending message...",
                searching_manuals: "Searching manuals...",
                generating_answer: "Generating answer...",
                error_occurred: "An error occurred"
            },
            ar: {
                upload_documents: "رفع المستندات",
                chat: "محادثة",
                upload_pdf_manuals: "رفع كتيبات PDF",
                drag_drop_files: "اسحب وأفلت ملفات PDF هنا",
                or_click_to_browse: "أو انقر للتصفح",
                choose_files: "اختر الملفات",
                force_reingest: "فرض إعادة المعالجة (استبدال المستندات الموجودة)",
                upload_documents: "رفع المستندات",
                manual_assistant: "مساعد الدليل",
                temperature: "درجة الحرارة",
                type_your_question: "اكتب سؤالك...",
                sources: "المصادر",
                welcome_message: "مرحباً! أنا هنا لمساعدتك في الأسئلة المتعلقة بالأدلة المرفوعة. يرجى رفع بعض مستندات PDF أولاً، ثم اسألني أي شيء!",
                processing: "معالجة...",
                uploading: "رفع الملفات...",
                file_uploaded: "تم رفع الملف بنجاح",
                file_failed: "فشل في رفع الملف",
                file_skipped: "تم تخطي الملف (موجود بالفعل)",
                no_files_selected: "لم يتم اختيار ملفات",
                upload_error: "خطأ في الرفع",
                connection_error: "خطأ في الاتصال",
                sending_message: "إرسال الرسالة...",
                searching_manuals: "البحث في الأدلة...",
                generating_answer: "توليد الإجابة...",
                error_occurred: "حدث خطأ"
            }
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadUserPreferences();
        this.updateTranslations();
        this.setupWebSocket();
    }
    
    setupEventListeners() {
        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });
        
        // Language toggle
        document.getElementById('languageToggle').addEventListener('click', () => {
            this.toggleLanguage();
        });
        
        // File upload
        const fileInput = document.getElementById('fileInput');
        const browseButton = document.getElementById('browseButton');
        const uploadArea = document.getElementById('uploadArea');
        const uploadButton = document.getElementById('uploadButton');
        
        browseButton.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files);
        });
        
        uploadButton.addEventListener('click', () => {
            this.uploadFiles();
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelection(e.dataTransfer.files);
        });
        
        // Chat
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        
        sendButton.addEventListener('click', () => {
            this.sendMessage();
        });
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Temperature slider
        const temperatureSlider = document.getElementById('temperatureSlider');
        const temperatureValue = document.getElementById('temperatureValue');
        
        temperatureSlider.addEventListener('input', (e) => {
            temperatureValue.textContent = e.target.value;
        });
    }
    
    loadUserPreferences() {
        // Load theme
        const savedTheme = localStorage.getItem('chatbot-theme') || 'light';
        this.setTheme(savedTheme);
        
        // Load language
        const savedLanguage = localStorage.getItem('chatbot-language') || 'en';
        this.setLanguage(savedLanguage);
    }
    
    toggleTheme() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }
    
    setTheme(theme) {
        this.currentTheme = theme;
        document.documentElement.setAttribute('data-bs-theme', theme);
        
        const themeIcon = document.getElementById('themeIcon');
        themeIcon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        
        localStorage.setItem('chatbot-theme', theme);
    }
    
    toggleLanguage() {
        const newLanguage = this.currentLanguage === 'en' ? 'ar' : 'en';
        this.setLanguage(newLanguage);
    }
    
    setLanguage(language) {
        this.currentLanguage = language;
        
        // Update document direction
        document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr';
        document.documentElement.lang = language;
        
        // Update language toggle button
        const languageText = document.getElementById('languageText');
        languageText.textContent = language === 'en' ? 'عربي' : 'English';
        
        localStorage.setItem('chatbot-language', language);
        this.updateTranslations();
    }
    
    updateTranslations() {
        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(element => {
            const key = element.getAttribute('data-i18n');
            if (this.translations[this.currentLanguage][key]) {
                element.textContent = this.translations[this.currentLanguage][key];
            }
        });
        
        // Update placeholders
        const placeholderElements = document.querySelectorAll('[data-i18n-placeholder]');
        placeholderElements.forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            if (this.translations[this.currentLanguage][key]) {
                element.placeholder = this.translations[this.currentLanguage][key];
            }
        });
    }
    
    translate(key) {
        return this.translations[this.currentLanguage][key] || key;
    }
    
    handleFileSelection(files) {
        this.selectedFiles = Array.from(files).filter(file => file.type === 'application/pdf');
        
        const uploadButton = document.getElementById('uploadButton');
        uploadButton.disabled = this.selectedFiles.length === 0;
        
        // Update upload area with file info
        this.updateUploadArea();
    }
    
    updateUploadArea() {
        const uploadArea = document.getElementById('uploadArea');
        
        if (this.selectedFiles.length === 0) {
            uploadArea.innerHTML = `
                <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                <h6 data-i18n="drag_drop_files">${this.translate('drag_drop_files')}</h6>
                <p class="text-muted mb-3">
                    <span data-i18n="or_click_to_browse">${this.translate('or_click_to_browse')}</span>
                </p>
                <button type="button" class="btn btn-outline-primary" id="browseButton">
                    <i class="fas fa-folder-open me-2"></i>
                    <span data-i18n="choose_files">${this.translate('choose_files')}</span>
                </button>
            `;
            
            // Re-attach event listener
            document.getElementById('browseButton').addEventListener('click', () => {
                document.getElementById('fileInput').click();
            });
        } else {
            const fileList = this.selectedFiles.map(file => `
                <div class="d-flex align-items-center mb-2">
                    <i class="fas fa-file-pdf text-danger me-2"></i>
                    <span class="flex-grow-1">${file.name}</span>
                    <span class="badge bg-secondary">${this.formatFileSize(file.size)}</span>
                </div>
            `).join('');
            
            uploadArea.innerHTML = `
                <div class="text-start">
                    <h6 class="mb-3">${this.selectedFiles.length} file(s) selected:</h6>
                    ${fileList}
                </div>
            `;
        }
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    async uploadFiles() {
        if (this.selectedFiles.length === 0 || this.isUploading) return;
        
        this.isUploading = true;
        const uploadButton = document.getElementById('uploadButton');
        const uploadProgress = document.getElementById('uploadProgress');
        const uploadStatus = document.getElementById('uploadStatus');
        const progressBar = uploadProgress.querySelector('.progress-bar');
        
        uploadButton.disabled = true;
        uploadProgress.style.display = 'block';
        uploadStatus.textContent = this.translate('uploading');
        progressBar.style.width = '0%';
        
        try {
            const formData = new FormData();
            this.selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            const forceReingest = document.getElementById('forceReingest').checked;
            formData.append('force', forceReingest);
            
            const response = await fetch('/api/ingest', {
                method: 'POST',
                body: formData
            });
            
            progressBar.style.width = '100%';
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.displayUploadResults(result);
            
            // Clear selected files
            this.selectedFiles = [];
            document.getElementById('fileInput').value = '';
            this.updateUploadArea();
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError(this.translate('upload_error'), error.message);
        } finally {
            this.isUploading = false;
            uploadButton.disabled = false;
            uploadProgress.style.display = 'none';
        }
    }
    
    displayUploadResults(result) {
        const resultsContainer = document.getElementById('uploadResults');
        
        const resultsHtml = result.documents.map(doc => {
            let statusClass = '';
            let statusIcon = '';
            let statusText = '';
            
            switch (doc.status) {
                case 'success':
                    statusClass = 'result-success';
                    statusIcon = 'fas fa-check-circle text-success';
                    statusText = this.translate('file_uploaded');
                    break;
                case 'failed':
                    statusClass = 'result-error';
                    statusIcon = 'fas fa-times-circle text-danger';
                    statusText = this.translate('file_failed');
                    break;
                case 'skipped':
                    statusClass = 'result-skipped';
                    statusIcon = 'fas fa-exclamation-triangle text-warning';
                    statusText = this.translate('file_skipped');
                    break;
            }
            
            return `
                <div class="result-item ${statusClass} fade-in">
                    <div class="d-flex align-items-start">
                        <i class="${statusIcon} me-2 mt-1"></i>
                        <div class="flex-grow-1">
                            <h6 class="mb-1">${doc.filename}</h6>
                            <p class="mb-1 small">${statusText}</p>
                            ${doc.status === 'success' ? 
                                `<small class="text-muted">Pages: ${doc.pages_processed}, Chunks: ${doc.chunks_created}</small>` : 
                                doc.error ? `<small class="text-danger">${doc.error}</small>` : ''
                            }
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        resultsContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                ${result.message}
            </div>
            ${resultsHtml}
        `;
    }
    
    setupWebSocket() {
        // WebSocket will be created when needed during chat
    }
    
    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return;
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError(this.translate('connection_error'), 'Failed to connect to chat service');
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.websocket = null;
        };
    }
    
    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message || this.isChatting) return;
        
        this.isChatting = true;
        messageInput.value = '';
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Show typing indicator
        const typingMessage = this.addMessage('', 'bot', true);
        
        try {
            this.connectWebSocket();
            
            // Wait for WebSocket to be ready
            if (this.websocket.readyState !== WebSocket.OPEN) {
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('Connection timeout')), 5000);
                    this.websocket.onopen = () => {
                        clearTimeout(timeout);
                        resolve();
                    };
                    this.websocket.onerror = () => {
                        clearTimeout(timeout);
                        reject(new Error('Connection failed'));
                    };
                });
            }
            
            // Send message
            const temperature = parseFloat(document.getElementById('temperatureSlider').value);
            this.websocket.send(JSON.stringify({
                message: message,
                session_id: 'default',
                temperature: temperature
            }));
            
        } catch (error) {
            console.error('Chat error:', error);
            this.removeMessage(typingMessage);
            this.addMessage(this.translate('error_occurred') + ': ' + error.message, 'bot');
            this.isChatting = false;
        }
    }
    
    handleWebSocketMessage(data) {
        const typingMessages = document.querySelectorAll('.message-typing');
        
        switch (data.type) {
            case 'status':
                // Update typing indicator with status
                if (typingMessages.length > 0) {
                    typingMessages[0].querySelector('.typing-status').textContent = data.message;
                }
                break;
                
            case 'sources':
                // Display sources
                this.displaySources(data.sources);
                break;
                
            case 'token':
                // Handle streaming tokens
                let streamingMessage = document.querySelector('.streaming-message');
                if (!streamingMessage) {
                    // Remove typing indicator and start streaming message
                    typingMessages.forEach(msg => msg.remove());
                    streamingMessage = this.addMessage('', 'bot', false, true);
                }
                
                const content = streamingMessage.querySelector('.message-content');
                content.innerHTML += data.content;
                this.scrollToBottom();
                break;
                
            case 'final_response':
                // Remove typing indicator
                typingMessages.forEach(msg => msg.remove());
                
                // Update or add final message
                let finalMessage = document.querySelector('.streaming-message');
                if (finalMessage) {
                    finalMessage.classList.remove('streaming-message');
                } else {
                    this.addMessage(data.answer, 'bot');
                }
                
                // Display sources
                this.displaySources(data.sources);
                this.isChatting = false;
                break;
                
            case 'error':
                // Remove typing indicator and show error
                typingMessages.forEach(msg => msg.remove());
                this.addMessage(this.translate('error_occurred') + ': ' + data.message, 'bot');
                this.isChatting = false;
                break;
        }
    }
    
    addMessage(content, sender, isTyping = false, isStreaming = false) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        
        let messageClass = sender === 'user' ? 'user-message' : 'bot-message';
        if (isTyping) messageClass += ' message-typing';
        if (isStreaming) messageClass += ' streaming-message';
        
        messageDiv.className = `message ${messageClass} fade-in`;
        
        if (isTyping) {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <i class="fas fa-robot me-2"></i>
                    <span class="typing-status">${this.translate('processing')}</span>
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
        } else {
            const icon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${sender === 'bot' ? `<i class="${icon} me-2"></i>` : ''}
                    ${content}
                </div>
            `;
        }
        
        chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    removeMessage(messageElement) {
        if (messageElement && messageElement.parentNode) {
            messageElement.parentNode.removeChild(messageElement);
        }
    }
    
    displaySources(sources) {
        if (!sources || sources.length === 0) {
            document.getElementById('sourcesPanel').style.display = 'none';
            return;
        }
        
        const sourcesPanel = document.getElementById('sourcesPanel');
        const sourcesList = document.getElementById('sourcesList');
        
        // Group sources by filename to avoid duplicates
        const uniqueSources = [];
        const seenSources = new Set();
        
        sources.forEach(source => {
            const key = `${source.filename}-${source.page}`;
            if (!seenSources.has(key)) {
                uniqueSources.push({
                    filename: source.filename,
                    page: source.page
                });
                seenSources.add(key);
            }
        });
        
        const sourcesHtml = uniqueSources.map(source => `
            <div class="source-item mb-2">
                <div class="d-flex align-items-center">
                    <i class="fas fa-file-pdf text-danger me-2"></i>
                    <strong>${source.filename}</strong>
                </div>
                <div class="text-muted ms-3">
                    <i class="fas fa-bookmark me-1"></i>
                    Page ${source.page}
                </div>
            </div>
        `).join('');
        
        sourcesList.innerHTML = sourcesHtml;
        sourcesPanel.style.display = 'block';
    }
    
    scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    showError(title, message) {
        // Simple error display - could be enhanced with a modal
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alertDiv.innerHTML = `
            <strong>${title}</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.container').firstChild);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Global function for source preview
function showSourcePreview(filename, page, snippet) {
    // Simple preview - could be enhanced with a modal showing more context
    alert(`Source: ${filename} (Page ${page})\n\n${snippet}`);
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatbotApp = new ChatbotApp();
});
