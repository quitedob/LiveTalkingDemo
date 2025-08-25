var pc = null;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        document.getElementById('sessionid').value = answer.sessionid
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        console.error('WebRTC协商失败:', e);
        // 不要弹出alert，只记录错误
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.miwifi.com:3478'] }];
    }

    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    // 添加连接状态监听
    pc.addEventListener('connectionstatechange', () => {
        console.log('WebRTC连接状态:', pc.connectionState);
        
        // 更新连接状态显示
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus) {
            connectionStatus.textContent = `WebRTC状态: ${pc.connectionState}`;
        }
        
        // 处理连接状态变化 - 移除自动断开逻辑
        switch (pc.connectionState) {
            case 'connecting':
                console.log('WebRTC正在连接...');
                break;
            case 'connected':
                console.log('WebRTC连接已建立');
                break;
            case 'disconnected':
                console.log('WebRTC连接断开，请手动重连或等待自动恢复...');
                // 不再自动处理，等待用户手动操作或连接自然恢复
                break;
            case 'failed':
                console.log('WebRTC连接失败，请手动重连');
                // 不再自动处理，等待用户手动操作
                break;
            case 'closed':
                console.log('WebRTC连接已关闭');
                break;
        }
    });

    // 添加ICE连接状态监听
    pc.addEventListener('iceconnectionstatechange', () => {
        console.log('ICE连接状态:', pc.iceConnectionState);
    });

    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}

window.onunload = function(event) {
    // 在这里执行你想要的操作
    setTimeout(() => {
        pc.close();
    }, 500);
};

window.onbeforeunload = function (e) {
        setTimeout(() => {
                pc.close();
            }, 500);
        e = e || window.event
        // 兼容IE8和Firefox 4之前的版本
        if (e) {
          e.returnValue = '关闭提示'
        }
        // Chrome, Safari, Firefox 4+, Opera 12+ , IE 9+
        return '关闭提示'
      }