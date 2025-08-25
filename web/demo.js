// demo.js

// 智能语音助手控制面板JavaScript文件

// 获取DOM元素
const settingsBtn = document.getElementById('settings-btn');
const closePanelBtn = document.getElementById('close-panel');
const settingsPanel = document.getElementById('settings-panel');
const toggleSubtitleBtn = document.getElementById('toggle-subtitle-btn');
const leftText = document.getElementById('left-text');
const rightText = document.getElementById('right-text');
const micBtn = document.getElementById('mic-btn');
const micAnimation = document.getElementById('mic-animation');
const startBtn = document.getElementById('start-btn');
const bookBtn = document.getElementById('book-btn');
const notification = document.getElementById('notification');

// WebRTC和ASR相关元素
const sessionIdInput = document.getElementById('sessionid');
const audio = document.getElementById('audio');
const video = document.getElementById('video');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const connectionStatus = document.getElementById('connection-status');
const asrResult = document.getElementById('asr-result');
const asrText = document.getElementById('asr-text');
const aiSubtitle = document.getElementById('ai-subtitle');
const userSubtitle = document.getElementById('user-subtitle');

// 新增：即时交互设置元素
const instantSettings = document.getElementById('instant-settings');
const instantRagSwitch = document.getElementById('instant-rag-switch');
const instantKbSelect = document.getElementById('instant-kb-select');
const instantVoiceSelect = document.getElementById('instant-voice-select');

// 同步音色选择
if (voiceListSelect && instantVoiceSelect) {
    voiceListSelect.addEventListener('change', () => {
        instantVoiceSelect.value = voiceListSelect.value;
    });
    instantVoiceSelect.addEventListener('change', () => {
        voiceListSelect.value = instantVoiceSelect.value;
    });
}

instantRagSwitch.addEventListener('change', async () => {
    if (instantRagSwitch.checked) {
        if (!kbSelect.value) {
            if (kbSelect.options.length > 1 && kbSelect.options[1].value) {
                const firstKb = kbSelect.options[1].value;
                await switchKb(firstKb);
            } else {
                showNotification('没有可用的知识库，无法开启RAG模式', true);
                instantRagSwitch.checked = false;
            }
        } else {
             showNotification('RAG模式已开启');
        }
    } else {
        if (kbSelect.value) {
            await switchKb('');
        } else {
            showNotification('RAG模式已关闭');
        }
    }
});

instantKbSelect.addEventListener('change', async () => {
    const selectedKb = instantKbSelect.value;
    if (selectedKb) {
        showNotification(`已切换到知识库: ${selectedKb}`);
    } else {
        showNotification('已取消使用知识库');
    }
});

// 新增：FishTTS相关元素
const uploadVoiceBtn = document.getElementById('upload-voice-btn');
const deleteVoiceBtn = document.getElementById('delete-voice-btn');
const refreshVoiceBtn = document.getElementById('refresh-voice-btn');
const voiceNameInput = document.getElementById('voice-name-input');
const voiceFileInput = document.getElementById('voice-file-input');
const voiceListSelect = document.getElementById('voice-list-select');
const voiceStatus = document.getElementById('voice-status');

// 新增：数字人管理相关元素
const createAvatarBtn = document.getElementById('create-avatar-btn');
const switchAvatarBtn = document.getElementById('switch-avatar-btn');
const deleteAvatarBtn = document.getElementById('delete-avatar-btn');
const avatarSelect = document.getElementById('avatar-select');
const avatarFileInput = document.getElementById('avatar-file-input');
const avatarIdInput = document.getElementById('avatar-id-input');
const avatarDeleteSelect = document.getElementById('avatar-delete-select');
const avatarStatus = document.getElementById('avatar-status');

// 新增：LLM管理相关元素
const switchLlmBtn = document.getElementById('switch-llm-btn');
const testLlmBtn = document.getElementById('test-llm-btn');
const refreshLlmBtn = document.getElementById('refresh-llm-btn');
const llmProviderSelect = document.getElementById('llm-provider-select');
const llmStatus = document.getElementById('llm-status');

// 状态管理对象
let state = {
    sessionId: null,
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    isConnected: false,
    heartbeatInterval: null,
    currentStream: null,
    // 新增：重连相关状态
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    isReconnecting: false,
    subtitlesVisible: true, // 新增：字幕显示状态
    // 新增：心跳相关状态
    lastHeartbeat: null,
    heartbeatFailures: 0,
    maxHeartbeatFailures: 3
};


// ===================================
// ★★★ [修复] 新增：为兼容 client.js 注入所需的 DOM 元素 ★★★
// 目标：解决 client.js 脚本因找不到 #start 和 #stop 元素而报错的问题。
// 原理：client.js 脚本硬编码依赖 id="start" 和 id="stop" 的按钮来管理其显隐状态。
//       demo.html 使用的是单个 id="start-btn" 按钮，缺少这两个元素，导致 client.js 内部
//       调用 document.getElementById('start') 时返回 null，从而引发 .style 访问错误。
//       此函数动态创建这两个 client.js 需要的虚拟按钮，并将其隐藏，从而避免脚本错误，
//       同时不影响 demo.html 自身的 UI 逻辑。
function ensureClientJsDom() {
    // 检查并创建 id="start" 按钮
    if (!document.getElementById('start')) {
        const dummyStart = document.createElement('button');
        dummyStart.id = 'start';
        dummyStart.style.display = 'none'; // 保持隐藏
        document.body.appendChild(dummyStart);
        console.log('为 client.js 兼容性创建虚拟 #start 按钮');
    }
    // 检查并创建 id="stop" 按钮
    if (!document.getElementById('stop')) {
        const dummyStop = document.createElement('button');
        dummyStop.id = 'stop';
        dummyStop.style.display = 'none'; // 保持隐藏
        document.body.appendChild(dummyStop);
        console.log('为 client.js 兼容性创建虚拟 #stop 按钮');
    }
}
// 在脚本加载时立即执行，确保在调用 window.start() 或 window.stop() 之前元素已存在
ensureClientJsDom();
// ===================================


// 设置面板开关
settingsBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('设置按钮被点击');
    
    settingsPanel.classList.toggle('active');
    settingsBtn.classList.toggle('active');
    
    // 如果打开设置面板，加载知识库列表并同步
    if (settingsPanel.classList.contains('active')) {
        loadKnowledgeBases().then(() => {
            syncKbToInstantSettings(); // 同步知识库
        });
    }
});

// 关闭设置面板
closePanelBtn.addEventListener('click', () => {
    settingsPanel.classList.remove('active');
    settingsBtn.classList.remove('active');
});

// 知识库管理按钮事件
document.getElementById('create-kb-btn').addEventListener('click', createKnowledgeBase);
document.getElementById('delete-kb-btn').addEventListener('click', deleteKnowledgeBase);
document.getElementById('switch-kb-btn').addEventListener('click', switchKnowledgeBase);

// 知识库选择变化时自动切换
document.getElementById('kb-select').addEventListener('change', switchKnowledgeBase);

// RAG提示词更新
document.getElementById('rag-prompt').addEventListener('blur', updateSystemPrompt);

// 字幕显示切换功能
toggleSubtitleBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    state.subtitlesVisible = !state.subtitlesVisible;

    if (state.subtitlesVisible) {
        leftText.style.display = 'block';
        rightText.style.display = 'block';
        toggleSubtitleBtn.classList.add('active');
        showNotification('字幕已显示');
    } else {
        leftText.style.display = 'none';
        rightText.style.display = 'none';
        toggleSubtitleBtn.classList.remove('active');
        showNotification('字幕已隐藏');
    }
});

// 文本输入功能
// textBtn.addEventListener('click', (e) => {
//     e.preventDefault();
//     e.stopPropagation();
    
//     console.log('文本按钮被点击');
    
//     if (!state.isConnected || !state.sessionId) {
//         showNotification('请先点击开始按钮连接WebRTC');
//         return;
//     }
    
//     // 弹出文本输入对话框
//     const userText = prompt('请输入要发送的文本:');
//     if (userText && userText.trim()) {
//         console.log('用户输入文本:', userText);
//         sendTextToAI(userText.trim());
//     }
// });

// 发送文本到AI
async function sendTextToAI(text) {
    if (!state.sessionId) {
        showNotification('请先建立WebRTC连接');
        return;
    }
    
    // 显示用户输入
    showASRResult(text);
    
    // 先发送打断请求
    await api.interruptTalk(state.sessionId).catch(e => console.error('打断请求失败:', e));
    
    // 从即时设置获取RAG和音色配置
    const useRag = instantRagSwitch.checked;
    const kbName = instantKbSelect.value;
    const selectedVoice = instantVoiceSelect.value;
    
    const payload = {
        sessionid: state.sessionId,
        text: text,
        interrupt: false,
        use_rag: useRag,
        kb_name: useRag ? kbName : null,
        type: 'chat',
        tts_options: {}
    };
    
    // 如果选中了克隆音源，则添加到 tts_options
    if (selectedVoice) {
        payload.tts_options.voice_clone_name = selectedVoice;
    }
    
    try {
        const response = await api.sendHumanText(payload);
        
        // 处理流式响应
        if (response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let llmResponse = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                
                for (const line of lines) {
                    if (line.trim() === '') continue;
                    try {
                        const jsonData = JSON.parse(line);
                        
                        // 处理LLM流式响应
                        if (jsonData.response) {
                            llmResponse += jsonData.response;
                            showAIResponse(llmResponse);
                        }
                        
                        // 处理完整的LLM响应
                        if (jsonData.llm_response) {
                            llmResponse = jsonData.llm_response;
                            showAIResponse(llmResponse);
                        }
                    } catch (e) {
                        console.warn('无法解析的JSON行:', line);
                    }
                }
            }
            
            // 处理剩余buffer
            if (buffer.trim() !== '') {
                try {
                    const jsonData = JSON.parse(buffer);
                    if (jsonData.response) {
                        llmResponse += jsonData.response;
                        showAIResponse(llmResponse);
                    }
                    if (jsonData.llm_response) {
                        showAIResponse(jsonData.llm_response);
                    }
                } catch (e) {
                    console.warn('无法解析的JSON行 (buffer):', buffer);
                }
            }
        }
        
        showNotification('文本处理完成');
    } catch (e) {
        showNotification('文本处理失败');
        console.error('文本处理错误:', e);
    }
}

// 语音录制功能
async function startRecording() {
    if (!state.sessionId) {
        showNotification('请先连接WebRTC');
        return;
    }
    
    // 先发送打断请求
    await api.interruptTalk(state.sessionId).catch(e => console.error("打断请求失败:", e));

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.currentStream = stream; // 保存stream引用
        state.mediaRecorder = new MediaRecorder(stream);
        state.audioChunks = [];
        
        console.log('获取麦克风权限成功，开始录音');
        
        state.mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) {
                state.audioChunks.push(e.data);
                console.log('录音数据块大小:', e.data.size);
            }
        };
        
        state.mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/wav' });
            
            // 从即时设置获取RAG和音色配置
            const useRag = instantRagSwitch.checked;
            const kbName = instantKbSelect.value;
            const selectedVoice = instantVoiceSelect.value;
            
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            formData.append('sessionid', state.sessionId);
            formData.append('use_rag', String(useRag));
            if (useRag && kbName) {
                formData.append('kb_name', kbName);
            }
            
            // 新增：将 tts_options 作为 JSON 字符串添加
            const ttsOptions = {};
            if (selectedVoice) {
                ttsOptions.voice_clone_name = selectedVoice;
            }
            formData.append('tts_options', JSON.stringify(ttsOptions));
            
            try {
                const response = await api.sendHumanAudio(formData);
                
                // 处理流式响应
                if (response.body) {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    let asrResult = '';
                    let llmResponse = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();
                        
                        for (const line of lines) {
                            if (line.trim() === '') continue;
                            try {
                                const jsonData = JSON.parse(line);
                                
                                // 处理ASR结果
                                if (jsonData.asr_result) {
                                    asrResult = jsonData.asr_result;
                                    showASRResult(asrResult);
                                }
                                
                                // 处理LLM流式响应
                                if (jsonData.response) {
                                    llmResponse += jsonData.response;
                                    showAIResponse(llmResponse);
                                }
                                
                                // 处理完整的LLM响应
                                if (jsonData.llm_response) {
                                    llmResponse = jsonData.llm_response;
                                    showAIResponse(llmResponse);
                                }
                            } catch (e) {
                                console.warn('无法解析的JSON行:', line);
                            }
                        }
                    }
                    
                    // 处理剩余buffer
                    if (buffer.trim() !== '') {
                        try {
                            const jsonData = JSON.parse(buffer);
                            if (jsonData.asr_result) {
                                showASRResult(jsonData.asr_result);
                            }
                            if (jsonData.response) {
                                llmResponse += jsonData.response;
                                showAIResponse(llmResponse);
                            }
                            if (jsonData.llm_response) {
                                showAIResponse(jsonData.llm_response);
                            }
                        } catch (e) {
                            console.warn('无法解析的JSON行 (buffer):', buffer);
                        }
                    }
                }
                
                showNotification('语音处理完成');
            } catch (e) {
                showNotification('语音处理失败');
                console.error('语音处理错误:', e);
            } finally {
                // 清理音频流
                if (state.currentStream) {
                    state.currentStream.getTracks().forEach(track => track.stop());
                    state.currentStream = null;
                }
            }
        };

        state.mediaRecorder.start();
        state.isRecording = true;

    } catch (e) {
        showNotification('无法访问麦克风');
    }
}

// 停止录音功能
function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.mediaRecorder.stop();
        state.isRecording = false;
        console.log('停止录音');
        
        // 清理音频流
        if (state.currentStream) {
            state.currentStream.getTracks().forEach(track => track.stop());
            state.currentStream = null;
        }
    }
}

// 麦克风按钮 - 修改为录音控制
micBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('麦克风按钮被点击，当前录音状态:', state.isRecording);
    console.log('当前连接状态:', state.isConnected, '会话ID:', state.sessionId);
    
    if (!state.isConnected || !state.sessionId) {
        showNotification('请先点击开始按钮连接WebRTC');
        return;
    }
    
    if (!state.isRecording) {
        startRecording();
        micBtn.classList.add('active');
        micAnimation.classList.add('active');
        showNotification('开始录音...');
    } else {
        stopRecording();
        micBtn.classList.remove('active');
        micAnimation.classList.remove('active');
        showNotification('录音结束，处理中...');
    }
});

// WebRTC连接功能
function connectWebRTC() {
    updateConnectionStatus('连接中...', '正在建立WebRTC连接');
    
    try {
        window.start();
        
        // 设置WebRTC监听器
        setTimeout(() => {
            setupWebRTCListeners();
        }, 500);
        
        // 优化连接状态监听，减少频繁检查
        let connectionCheckAttempts = 0;
        const maxConnectionCheckAttempts = 60; // 最多检查60次（30秒）
        
        const checkConnection = setInterval(() => {
            connectionCheckAttempts++;
            
            if (sessionIdInput.value && sessionIdInput.value !== "0") {
                clearInterval(checkConnection);
                state.sessionId = sessionIdInput.value;
                updateConnectionStatus('已连接', '语音助手已就绪');
                showNotification('WebRTC连接成功！');
                
                // 隐藏欢迎语
                document.querySelector('.welcome-text').style.display = 'none';

                // 显示字幕区域和即时设置面板
                leftText.classList.add('visible');
                rightText.classList.add('visible');
                instantSettings.classList.add('visible');
                
                // 延迟启动心跳，避免连接刚建立就发送心跳
                setTimeout(() => {
                    if (state.heartbeatInterval) clearInterval(state.heartbeatInterval);
                    state.heartbeatInterval = setInterval(() => {
                        if (state.sessionId) {
                            api.sessionHeartbeat(state.sessionId).catch(e => console.warn("心跳失败:", e));
                        }
                    }, 30000); // 30秒心跳间隔
                    console.log('心跳监控已启动');
                }, 5000); // 连接成功5秒后再启动心跳
                
                console.log('WebRTC连接成功，会话ID:', state.sessionId);
                
            } else if (connectionCheckAttempts >= maxConnectionCheckAttempts) {
                // 连接超时处理
                clearInterval(checkConnection);
                console.error('WebRTC连接超时');
                updateConnectionStatus('连接超时', '请重试连接');
                showNotification('连接超时，请重试');
                
                // 重置按钮状态
                const startBtn = document.getElementById('start-btn');
                if (startBtn) {
                    startBtn.innerHTML = '<i class="fas fa-play"></i>';
                    startBtn.disabled = false;
                }
            }
        }, 500); // 保持500ms检查间隔，但增加超时机制
        
    } catch (e) {
        updateConnectionStatus('连接失败', '无法建立WebRTC连接');
        showNotification('WebRTC连接失败');
        console.error('WebRTC连接错误:', e);
    }
}

// 断开WebRTC连接
function disconnectWebRTC() {
    // 停止心跳和重连
    if (state.heartbeatInterval) clearInterval(state.heartbeatInterval);
    state.heartbeatInterval = null;
    state.isReconnecting = false;
    
    if (window.stop) window.stop();
    state.sessionId = null;
    sessionIdInput.value = '';
    updateConnectionStatus('未连接', '语音助手已断开');
    showNotification('WebRTC连接已断开');
}

// 开始按钮 - 修改为WebRTC连接控制
startBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const isActive = startBtn.classList.contains('active');
    console.log('开始按钮被点击，当前状态:', isActive ? '已连接' : '未连接');
    
    if (!isActive) {
        // 开始连接
        console.log('开始建立WebRTC连接');
        connectWebRTC();
        startBtn.classList.add('active');
        startBtn.innerHTML = '<i class="fas fa-stop"></i>';
        startBtn.title = '断开连接';
    } else {
        // 断开连接
        console.log('断开WebRTC连接');
        disconnectWebRTC();
        startBtn.classList.remove('active');
        startBtn.innerHTML = '<i class="fas fa-play"></i>';
        startBtn.title = '开始连接';
        
        // 恢复欢迎语
        document.querySelector('.welcome-text').style.display = 'block';

        // 清空字幕
        clearSubtitles();
    }
});

// 知识库管理功能
async function loadKnowledgeBases() {
    try {
        const data = await api.getKbList();
        const kbSelect = document.getElementById('kb-select');
        kbSelect.innerHTML = '<option value="">无</option>';
        
        if (data.knowledge_bases && data.knowledge_bases.length > 0) {
            data.knowledge_bases.forEach(kb => {
                const option = document.createElement('option');
                option.value = kb;
                option.textContent = kb;
                kbSelect.appendChild(option);
            });
        }
        
        // 加载当前配置
        const config = await api.getConfig();
        if (config.current_kb) {
            kbSelect.value = config.current_kb;
        }
        document.getElementById('rag-prompt').value = config.system_prompt || '';
        
    } catch (e) {
        console.error('加载知识库失败:', e);
    }
}

// 创建知识库
async function createKnowledgeBase() {
    const kbName = document.getElementById('kb-name').value.trim();
    const file = document.getElementById('kb-file').files[0];
    
    if (!kbName.match(/^[a-zA-Z0-9]{3,20}$/)) {
        showNotification('知识库名称不合法 (3-20位英文/数字)');
        return;
    }
    if (!file) {
        showNotification('请选择要上传的文件');
        return;
    }
    
    const formData = new FormData();
    formData.append('kb_name', kbName);
    formData.append('file', file);
    
    try {
        await api.createKb(formData);
        showNotification('知识库创建成功');
        document.getElementById('kb-name').value = '';
        document.getElementById('kb-file').value = '';
        await loadKnowledgeBases();
    } catch (e) {
        showNotification('知识库创建失败');
    }
}

// 删除知识库
async function deleteKnowledgeBase() {
    const selectedKb = document.getElementById('kb-select').value;
    if (!selectedKb) {
        showNotification('请选择要删除的知识库');
        return;
    }
    
    if (confirm(`确定要删除知识库 "${selectedKb}" 吗？`)) {
        try {
            await api.deleteKb(selectedKb);
            showNotification('知识库删除成功');
            await loadKnowledgeBases();
        } catch (e) {
            showNotification('知识库删除失败');
        }
    }
}

// 切换知识库
async function switchKnowledgeBase() {
    const selectedKb = document.getElementById('kb-select').value;
    await setKnowledgeBase(selectedKb);
}

// 更新系统提示词
async function updateSystemPrompt() {
    const prompt = document.getElementById('rag-prompt').value;
    try {
        await api.updatePrompt(prompt);
        showNotification('提示词更新成功');
    } catch (e) {
        showNotification('提示词更新失败');
    }
}

/**
 * 设置知识库，并同步两个面板的UI状态
 * @param {string} kbName - 要切换的知识库名称，传空字符串则为关闭
 */
async function setKnowledgeBase(kbName) {
    const kbSelect = document.getElementById('kb-select');
    const instantKbSelect = document.getElementById('instant-kb-select');
    const instantRagSwitch = document.getElementById('instant-rag-switch');

    try {
        await api.switchKb(kbName);
        
        // 更新UI
        kbSelect.value = kbName;
        instantKbSelect.value = kbName;
        instantRagSwitch.checked = !!kbName;

        showNotification(kbName ? `已切换到知识库: ${kbName}` : '已取消使用知识库');
    } catch (e) {
        showNotification('知识库切换失败');
        console.error('知识库切换失败:', e);
        // 失败时从服务器重新加载状态以恢复UI
        await loadKnowledgeBases();
        syncKbToInstantSettings();
    }
}

/**
 * 绑定所有UI同步相关的事件监听器
 */
function setupSyncEventListeners() {
    const voiceListSelect = document.getElementById('voice-list-select');

    // 1. 知识库同步 (kb-select的监听器已在外部设置，并调用了更新后的switchKnowledgeBase)
    // 即时面板的知识库选择
    instantKbSelect.addEventListener('change', () => {
        setKnowledgeBase(instantKbSelect.value);
    });

    // 即时面板的RAG开关
    instantRagSwitch.addEventListener('change', async () => {
        const kbSelect = document.getElementById('kb-select');
        if (instantRagSwitch.checked) {
            // 如果开启时没有选中的知识库，则自动选择第一个
            if (!kbSelect.value) {
                if (kbSelect.options.length > 1 && kbSelect.options[1].value) {
                    const firstKb = kbSelect.options[1].value;
                    await setKnowledgeBase(firstKb);
                } else {
                    showNotification('没有可用的知识库，无法开启RAG模式', true);
                    instantRagSwitch.checked = false; // 恢复原状
                }
            }
        } else {
            // 如果关闭，则取消知识库选择
            if (kbSelect.value) {
                await setKnowledgeBase('');
            }
        }
    });

    // 2. 音色选择同步
    // 设置面板的音色选择
    voiceListSelect.addEventListener('change', () => {
        instantVoiceSelect.value = voiceListSelect.value;
    });

    // 即时面板的音色选择
    instantVoiceSelect.addEventListener('change', () => {
        voiceListSelect.value = instantVoiceSelect.value;
    });

    console.log('UI同步事件监听器已设置');
}

// 书本按钮 - 跳转页面
bookBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('书本按钮被点击');
    showNotification('正在跳转到webrtc1.html...');
    
    setTimeout(() => {
        window.location.href = 'webrtc1.html';
    }, 500);
});

// 显示通知
function showNotification(message) {
    notification.textContent = message;
    notification.classList.add('show');
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 2000);
}

// 更新连接状态
function updateConnectionStatus(status, message) {
    connectionStatus.textContent = `WebRTC状态: ${status}`;
    statusText.textContent = message;
    
    if (status === '已连接') {
        statusDot.style.backgroundColor = '#4caf50';
        state.isConnected = true;
    } else {
        statusDot.style.backgroundColor = '#f44336';
        state.isConnected = false;
    }
}

// 显示ASR结果（用户输入 - 黄色）
function showASRResult(text) {
    if (!text) return;
    
    asrText.textContent = text;
    asrResult.style.opacity = '1';
    userSubtitle.textContent = text; // 只更新气泡内的文本
    rightText.classList.add('visible');
    
    console.log('显示ASR结果:', text);
    
    // 5秒后隐藏ASR结果框
    setTimeout(() => {
        asrResult.style.opacity = '0';
    }, 5000);
}

// 显示AI回复（LLM响应 - 绿色）
function showAIResponse(text) {
    if (!text) return;
    
    aiSubtitle.textContent = `AI: ${text}`;
    leftText.classList.add('visible');
    
    console.log('显示AI回复:', text);
}

// 清空字幕
function clearSubtitles() {
    aiSubtitle.textContent = 'AI回复将在这里显示...';
    userSubtitle.textContent = '用户语音识别结果将在这里显示...';
    leftText.classList.remove('visible');
    rightText.classList.remove('visible');
    instantSettings.classList.remove('visible');
    asrResult.style.opacity = '0';
}

// API调用函数
const api = {
    async fetch(url, options = {}) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({error: '未知错误'}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            return response;
        } catch (error) {
            showNotification(`请求失败: ${error.message}`);
            console.error('API Error:', error);
            throw error;
        }
    },
    
    sendHumanAudio: (formData) => api.fetch('/audio_chat', { method: 'POST', body: formData }),
    sendHumanText: (payload) => api.fetch('/human', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) }),
    interruptTalk: (sessionId) => api.fetch('/interrupt_talk', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sessionid: sessionId }) }),
    sessionHeartbeat: (sessionId) => api.fetch('/session/heartbeat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sessionid: Number(sessionId) }) }),
    
    // 知识库管理API
    getKbList: () => api.fetch('/kb/list').then(res => res.json()),
    createKb: (formData) => api.fetch('/kb/create', { method: 'POST', body: formData }),
    deleteKb: (name) => api.fetch(`/kb/delete/${name}`, { method: 'DELETE' }),
    switchKb: (name) => api.fetch(`/kb/switch/${name}`, { method: 'POST' }),
    getConfig: () => api.fetch('/config/get').then(res => res.json()),
    updatePrompt: (prompt) => api.fetch('/config/prompt', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt}) }),
    
    // 新增：重连API
    reconnect: (payload) => api.fetch('/reconnect', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) }),
    
    // 新增：数字人管理API
    getAvatars: (page = 1, pageSize = 20) => api.fetch(`/api/avatars?page=${page}&page_size=${pageSize}`).then(res => res.json()),
    createAvatar: (formData) => api.fetch('/api/avatars', { method: 'POST', body: formData }).then(res => res.json()),
    deleteAvatar: (id) => api.fetch(`/api/avatars/${id}`, { method: 'DELETE' }),
    switchAvatar: (sessionId, avatarId) => api.fetch(`/api/sessions/${sessionId}/avatar`, { 
        method: 'PATCH', 
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ avatar_id: avatarId }) 
    }).then(res => res.json()),
    getAvatarTaskStatus: (jobId) => api.fetch(`/api/avatars/tasks/${jobId}`).then(res => res.json()),
    
    // 新增：FishTTS声音克隆API
    getFishttsVoices: () => api.fetch('/fishtts/voices').then(res => res.json()),
    uploadFishttsVoice: (formData) => api.fetch('/fishtts/voices', { method: 'POST', body: formData }).then(res => res.json()),
    deleteFishttsVoice: (name) => api.fetch(`/fishtts/voices/${name}`, { method: 'DELETE' }),
    
    // 新增：RAG聊天API
    ragChat: (query) => api.fetch('/rag/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query}) }),
    setRagMode: (use_rag) => api.fetch('/config/rag_mode', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({use_rag}) }),

    // 新增：LLM管理API
    getLlmProviders: () => api.fetch('/llm/providers').then(res => res.json()),
    switchLlmProvider: (provider) => api.fetch('/llm/switch', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({provider}) }).then(res => res.json()),
    testLlm: (query = 'Hello, please respond with a simple greeting.') => api.fetch('/llm/test', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query}) }).then(res => res.json()),
    
    // 新增：会话管理API
    closeSession: (sessionId) => api.fetch('/session/close', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sessionid: sessionId }) })
};

// WebRTC连接状态监听
function setupWebRTCListeners() {
    if (typeof pc !== 'undefined' && pc) {
        pc.addEventListener('connectionstatechange', handleConnectionStateChange);
        
        pc.addEventListener('track', (evt) => {
            if (evt.track.kind == 'video') {
                video.srcObject = evt.streams[0];
            } else {
                audio.srcObject = evt.streams[0];
            }
        });
    }
}

// 新增：连接状态变化处理 - 移除自动断开逻辑
function handleConnectionStateChange() {
    if (!pc) return;
    console.log(`WebRTC连接状态变更: ${pc.connectionState}`);
    
    switch (pc.connectionState) {
        case 'connecting':
            updateConnectionStatus('连接中', '正在建立连接...');
            break;
        case 'connected':
            if (state.isReconnecting) {
                showNotification('WebRTC连接已恢复');
                updateConnectionStatus('已连接', '语音助手已就绪');
                state.isReconnecting = false;
                state.reconnectAttempts = 0;
            } else {
                updateConnectionStatus('已连接', '语音助手已就绪');
            }
            state.isConnected = true;
            startHeartbeat(); // 连接成功后开始心跳
            break;
        case 'disconnected':
            console.log('WebRTC连接断开，系统保留会话等待重连或用户操作...');
            updateConnectionStatus('连接断开', '系统保留会话5分钟，可手动重连');
            showNotification('连接断开，系统保留会话5分钟，可重新点击开始按钮', true);
            state.isConnected = false;
            // 不再自动处理，等待用户手动操作
            break;
        case 'failed':
            console.log('WebRTC连接失败，等待用户手动重连');
            updateConnectionStatus('连接失败', '请手动重新连接');
            showNotification('连接失败，请点击开始按钮重新连接', true);
            state.isConnected = false;
            stopHeartbeat(); // 停止心跳
            // 不再自动断开，等待用户手动操作
            break;
        case 'closed':
            console.log('WebRTC连接已关闭');
            state.isConnected = false;
            stopHeartbeat(); // 停止心跳
            break;
        default:
            updateConnectionStatus(pc.connectionState, `连接状态: ${pc.connectionState}`);
    }
}

// 新增：ICE重启重连逻辑
async function tryReconnect() {
    if (!pc || !state.sessionId) return;
    
    console.log('尝试进行ICE重启...');
    
    try {
        const offer = await pc.createOffer({ iceRestart: true });
        await pc.setLocalDescription(offer);
        
        const response = await api.reconnect({
            sessionid: Number(state.sessionId),
            sdp: offer.sdp,
            type: offer.type
        });
        
        if (!response.ok) {
            throw new Error(`重连请求失败: ${response.status}`);
        }
        
        const ans = await response.json();
        await pc.setRemoteDescription({ type: ans.type, sdp: ans.sdp });
        console.log('ICE重启offer/answer交换完成');
        
    } catch (error) {
        console.error('重连过程中发生错误:', error);
        throw error;
    }
}

// 优化：断线重连处理，移除激进重连 - 改为手动重连模式
// 注释掉自动重连逻辑，等待用户手动操作
/*
async function handleDisconnect() {
    // 避免重复重连
    if (state.isReconnecting) {
        console.log('重连已在进行中，跳过');
        return;
    }
    
    state.isReconnecting = true;
    state.reconnectAttempts = 0;
    updateConnectionStatus('连接断开', '等待重连...');
    
    console.log('检测到连接断开，等待一段时间观察连接状态...');
    
    // 给连接一些时间自然恢复
    await new Promise(r => setTimeout(r, 3000));
    
    // 检查连接是否已经自然恢复
    if (pc && (pc.connectionState === 'connected' || pc.connectionState === 'connecting')) {
        console.log('连接已自然恢复，取消重连');
        state.isReconnecting = false;
        updateConnectionStatus('已连接', '语音助手已就绪');
        return;
    }

    while (state.reconnectAttempts < state.maxReconnectAttempts && state.isReconnecting) {
        state.reconnectAttempts++;
        console.log(`开始第 ${state.reconnectAttempts}/${state.maxReconnectAttempts} 次重连尝试`);
        updateConnectionStatus('重连中', `尝试重连 (${state.reconnectAttempts}/${state.maxReconnectAttempts})`);
        showNotification(`正在重连 (${state.reconnectAttempts}/${state.maxReconnectAttempts})...`);

        try {
            await tryReconnect();
            // 等待连接状态稳定
            await new Promise(r => setTimeout(r, 5000));

            if (pc && pc.connectionState === 'connected') {
                console.log('重连成功！');
                updateConnectionStatus('已连接', '连接已恢复');
                showNotification('连接已恢复');
                state.isReconnecting = false;
                return;
            }
        } catch (error) {
            console.error(`重连尝试 ${state.reconnectAttempts} 失败:`, error);
            // 递增等待时间，避免过于频繁的重连
            await new Promise(r => setTimeout(r, 3000 * state.reconnectAttempts));
        }
    }

    if (state.isReconnecting) {
        showNotification('重连失败，请手动重新连接');
        disconnectWebRTC();
    }
}
*/

// 心跳机制
function startHeartbeat() {
    if (state.heartbeatInterval) {
        clearInterval(state.heartbeatInterval);
    }
    
    state.heartbeatInterval = setInterval(async () => {
        if (state.sessionId) {
            try {
                await api.sessionHeartbeat(Number(state.sessionId));
                state.lastHeartbeat = Date.now();
                state.heartbeatFailures = 0;
                console.log('心跳发送成功');
            } catch (error) {
                state.heartbeatFailures++;
                console.error(`心跳失败 (${state.heartbeatFailures}/${state.maxHeartbeatFailures}):`, error);
                
                if (state.heartbeatFailures >= state.maxHeartbeatFailures) {
                    console.warn('心跳连续失败，可能需要重连');
                    showNotification('连接不稳定，正在检查...', true);
                }
            }
        }
    }, 30000); // 每30秒发送一次心跳
}

function stopHeartbeat() {
    if (state.heartbeatInterval) {
        clearInterval(state.heartbeatInterval);
        state.heartbeatInterval = null;
    }
}

// 初始化检查
setTimeout(() => {
    setupWebRTCListeners();
}, 1000);

// 新增：加载知识库列表到即时设置
async function loadKnowledgeBasesToInstant() {
    try {
        const data = await api.getKbList();
        instantKbSelect.innerHTML = '<option value="">无</option>';
        
        if (data.knowledge_bases && data.knowledge_bases.length > 0) {
            data.knowledge_bases.forEach(kb => {
                const option = document.createElement('option');
                option.value = kb;
                option.textContent = kb;
                instantKbSelect.appendChild(option);
            });
        }
        
        // 加载当前配置
        const config = await api.getConfig();
        if (config.current_kb) {
            instantKbSelect.value = config.current_kb;
        }
        instantRagSwitch.checked = config.use_rag || false;
        
    } catch (e) {
        console.error('加载知识库失败:', e);
    }
}

// 新增：加载FishTTS音源列表到即时设置
async function loadFishttsVoicesToInstant() {
    try {
        const data = await api.getFishttsVoices();
        instantVoiceSelect.innerHTML = '<option value="">默认音色</option>';
        
        if (data.voices && data.voices.length > 0) {
            data.voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice;
                option.textContent = voice;
                instantVoiceSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error("加载FishTTS音源失败:", error);
    }
}

// 新增：FishTTS相关功能函数
async function loadFishttsVoices() {
    try {
        const data = await api.getFishttsVoices();
        
        // 更新即时音色选择和设置面板的音色列表
        const updateVoiceSelects = (voices) => {
            const options = '<option value="">默认音色</option>' + 
                           voices.map(voice => `<option value="${voice}">${voice}</option>`).join('');
            instantVoiceSelect.innerHTML = options;
            voiceListSelect.innerHTML = options;
        };
        
        if (data.voices && data.voices.length > 0) {
            updateVoiceSelects(data.voices);
            voiceStatus.textContent = `仓库中有 ${data.voices.length} 个音源。`;
        } else {
            updateVoiceSelects([]);
            voiceStatus.textContent = '音源仓库为空，将使用默认音色。';
        }
    } catch (error) {
        console.error("加载FishTTS音源失败:", error);
        voiceStatus.textContent = '加载音源列表失败。';
    }
}

async function handleUploadFishttsVoice() {
    const voiceName = voiceNameInput.value.trim();
    const file = voiceFileInput.files[0];

    if (!voiceName.match(/^[a-zA-Z0-9_]{3,20}$/)) {
        showNotification("名称不合法 (3-20位英文/数字/下划线)");
        return;
    }
    if (!file) {
        showNotification("请选择一个音频文件");
        return;
    }

    const formData = new FormData();
    formData.append('voice_name', voiceName);
    formData.append('audio_file', file);

    try {
        const data = await api.uploadFishttsVoice(formData);
        showNotification(`音源 ${data.voice_name} 上传成功!`);
        voiceNameInput.value = '';
        voiceFileInput.value = '';
        await loadFishttsVoices(); // 重新加载列表
    } catch (e) {
        console.error('音源上传失败:', e);
    }
}

async function handleDeleteFishttsVoice() {
    const selectedVoice = voiceListSelect.value;
    if (!selectedVoice) {
        showNotification("请先选择一个要删除的音源");
        return;
    }

    if (confirm(`确定要删除音源 "${selectedVoice}" 吗？此操作不可恢复。`)) {
        try {
            await api.deleteFishttsVoice(selectedVoice);
            showNotification(`音源 ${selectedVoice} 已删除`);
            await loadFishttsVoices(); // 重新加载列表
        } catch (e) {
            console.error('音源删除失败:', e);
        }
    }
}

// 新增：LLM管理功能函数
async function loadLlmProviders() {
    try {
        const data = await api.getLlmProviders();
        llmProviderSelect.value = data.current_provider;
        llmStatus.textContent = `当前: ${data.current_provider} (${data.client_info.model || 'N/A'})`;
    } catch (error) {
        console.error("加载LLM提供商失败:", error);
        llmStatus.textContent = '加载LLM状态失败';
    }
}

async function handleSwitchLlm() {
    const selectedProvider = llmProviderSelect.value;
    try {
        const result = await api.switchLlmProvider(selectedProvider);
        showNotification(`LLM已切换到 ${result.current_provider}`);
        llmStatus.textContent = `当前: ${result.current_provider} (${result.client_info.model || 'N/A'})`;
    } catch (error) {
        console.error("切换LLM失败:", error);
        showNotification("切换LLM失败", true);
    }
}

async function handleTestLlm() {
    try {
        showNotification('正在测试LLM连接...');
        const result = await api.testLlm();
        if (result.success) {
            showNotification(`LLM测试成功 (${result.provider})`);
        } else {
            showNotification(`LLM测试失败: ${result.error}`, true);
        }
    } catch (error) {
        console.error("测试LLM失败:", error);
        showNotification("测试LLM失败", true);
    }
}

// 新增：数字人管理功能函数
async function loadAvatars() {
    try {
        const data = await api.getAvatars();
        
        // 重置选择框
        avatarSelect.innerHTML = '<option value="">选择数字人</option>';
        avatarDeleteSelect.innerHTML = '<option value="">选择要删除的数字人</option>';

        if (data.avatars && data.avatars.length > 0) {
            data.avatars.forEach(avatar => {
                // 为选择框添加选项
                const selectOption = document.createElement('option');
                selectOption.value = avatar.avatar_id;
                selectOption.textContent = `${avatar.avatar_id} (${avatar.frames || 0}帧)`;
                avatarSelect.appendChild(selectOption);

                // 为删除框添加选项
                const deleteOption = document.createElement('option');
                deleteOption.value = avatar.avatar_id;
                deleteOption.textContent = `${avatar.avatar_id} (${avatar.frames || 0}帧)`;
                avatarDeleteSelect.appendChild(deleteOption);
            });
            avatarStatus.textContent = `共有 ${data.total || data.avatars.length} 个数字人。`;
        } else {
            avatarStatus.textContent = '暂无数字人，请先创建。';
        }
    } catch (error) {
        console.error("加载数字人列表失败:", error);
        avatarStatus.textContent = '加载数字人列表失败。';
    }
}

async function handleCreateAvatar() {
    const file = avatarFileInput.files[0];
    const avatarId = avatarIdInput.value.trim();

    if (!file) {
        showNotification("请选择一个文件");
        return;
    }

    // 验证avatar_id格式（如果提供）
    if (avatarId && !avatarId.match(/^[a-z]{1,16}\d{1,4}$/)) {
        showNotification("数字人ID格式不正确，应为1-16位拼音+1-4位数字，如：xiaoli0001");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    if (avatarId) {
        formData.append('avatar_id', avatarId);
    }

    try {
        const result = await api.createAvatar(formData);
        showNotification(`数字人创建任务已提交，任务ID: ${result.job_id}`);
        
        // 清空表单
        avatarFileInput.value = '';
        avatarIdInput.value = '';
        
        // 开始轮询任务状态
        pollAvatarTaskStatus(result.job_id, result.avatar_id);
        
    } catch (e) {
        console.error('数字人创建失败:', e);
    }
}

async function pollAvatarTaskStatus(jobId, avatarId) {
    const maxAttempts = 60; // 最多轮询60次（5分钟）
    let attempts = 0;
    
    const poll = async () => {
        try {
            const taskInfo = await api.getAvatarTaskStatus(jobId);
            
            if (taskInfo.status === 'SUCCEEDED') {
                showNotification(`数字人 ${avatarId} 创建成功！`);
                await loadAvatars(); // 重新加载列表
                return;
            } else if (taskInfo.status === 'FAILED') {
                showNotification(`数字人 ${avatarId} 创建失败: ${taskInfo.stderr_tail || '未知错误'}`);
                return;
            }
            
            // 继续轮询
            attempts++;
            if (attempts < maxAttempts) {
                setTimeout(poll, 5000); // 5秒后再次检查
            } else {
                showNotification(`数字人 ${avatarId} 创建超时，请稍后检查状态`);
            }
        } catch (error) {
            console.error("轮询任务状态失败:", error);
            attempts++;
            if (attempts < maxAttempts) {
                setTimeout(poll, 5000);
            }
        }
    };
    
    poll();
}

async function handleSwitchAvatar() {
    const selectedAvatar = avatarSelect.value;
    const sessionId = state.sessionId;

    if (!selectedAvatar) {
        showNotification("请先选择一个数字人");
        return;
    }
    if (!sessionId) {
        showNotification("请先建立WebRTC连接");
        return;
    }

    try {
        // 先"打断说话"
        await api.interruptTalk(sessionId);

        // 再请求切换
        const result = await api.switchAvatar(sessionId, selectedAvatar);

        showNotification(`数字人切换成功: ${result.previous || '无'} -> ${result.current}`);

        // 检查是否存在同名音色并自动切换
        const voiceExists = Array.from(instantVoiceSelect.options).some(option => option.value === selectedAvatar);
        if (voiceExists) {
            instantVoiceSelect.value = selectedAvatar;
            showNotification(`同时为您切换到同名音色: ${selectedAvatar}`);
        }

    } catch (e) {
        // 错误处理...
        if (e.message.includes("会话不存在")) {
            showNotification('会话失效，请重新建立连接。');
            disconnectWebRTC(); // 直接断开让用户重连
        } else {
            console.error('数字人切换失败:', e);
            showNotification('数字人切换失败，请查看控制台');
        }
    }
}

async function handleDeleteAvatar() {
    const selectedAvatar = avatarDeleteSelect.value;
    
    if (!selectedAvatar) {
        showNotification("请先选择一个要删除的数字人");
        return;
    }

    if (confirm(`确定要删除数字人 "${selectedAvatar}" 吗？此操作不可恢复。`)) {
        try {
            await api.deleteAvatar(selectedAvatar);
            showNotification(`数字人 ${selectedAvatar} 已删除`);
            await loadAvatars(); // 重新加载列表
        } catch (e) {
            console.error('数字人删除失败:', e);
        }
    }
}

// 页面初始化
function initializePage() {
    console.log('页面初始化开始');
    
    // 设置初始状态
    updateConnectionStatus('未连接', '语音助手待机中');
    clearSubtitles();
    
    // 设置按钮标题
    startBtn.title = '开始连接';
    micBtn.title = '语音录音';
    settingsBtn.title = '系统设置';
    toggleSubtitleBtn.title = '切换字幕显示';
    bookBtn.title = '跳转到webrtc1.html';
    
    // 加载即时设置数据
    loadKnowledgeBasesToInstant();
    loadFishttsVoices(); // 加载完整的FishTTS数据
    loadAvatars(); // 加载数字人数据
    loadLlmProviders(); // 加载LLM提供商数据
    
    // 新增：设置同步事件监听器
    setupSyncEventListeners();
    
    console.log('页面初始化完成');
}

// 新增：同步知识库到即时设置面板
function syncKbToInstantSettings() {
    const kbSelect = document.getElementById('kb-select');
    instantKbSelect.innerHTML = kbSelect.innerHTML;
    instantKbSelect.value = kbSelect.value;

    const ragConfig = document.getElementById('rag-prompt').value;
    // 这里可以根据需要决定是否同步rag prompt

    console.log("知识库列表已同步到即时设置");
}


// 模拟加载
setTimeout(() => {
    document.querySelector('.welcome-text').innerHTML = 
        '系统初始化完成<br>点击播放按钮连接WebRTC<br>点击麦克风按钮开始语音交互';
}, 2000);

// 页面加载完成后初始化
setTimeout(() => {
    initializePage();
}, 3000);

// 添加键盘快捷键支持
document.addEventListener('keydown', (e) => {
    // 空格键：开始/停止录音
    if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        if (state.isConnected) {
            if (!state.isRecording) {
                micBtn.click();
            }
        }
    }
    // Escape键：停止录音
    if (e.code === 'Escape' && state.isRecording) {
        micBtn.click();
    }
});

// 添加页面可见性检测
document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.isRecording) {
        // 页面隐藏时停止录音
        stopRecording();
        micBtn.classList.remove('active');
        micAnimation.classList.remove('active');
        showNotification('页面隐藏，录音已停止');
    }
});

// 新增：绑定FishTTS相关事件
uploadVoiceBtn.addEventListener('click', handleUploadFishttsVoice);
deleteVoiceBtn.addEventListener('click', handleDeleteFishttsVoice);
refreshVoiceBtn.addEventListener('click', loadFishttsVoices);

// 新增：绑定LLM相关事件
switchLlmBtn.addEventListener('click', handleSwitchLlm);
testLlmBtn.addEventListener('click', handleTestLlm);
refreshLlmBtn.addEventListener('click', loadLlmProviders);

// 新增：绑定数字人管理事件
createAvatarBtn.addEventListener('click', handleCreateAvatar);
switchAvatarBtn.addEventListener('click', handleSwitchAvatar);
deleteAvatarBtn.addEventListener('click', handleDeleteAvatar);