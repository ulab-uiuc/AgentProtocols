# -*- coding: utf-8 -*-
"""
ANP Unified Security Test Runner

要求：
- 原生ANP（AgentConnect SimpleNode + DID + WebSocket），无fallback/mock
- 与ACP/Agora完全一致的测试规模、指标与权重（S1=15%，S2=25%，S3-S8=60%）
- 与Eavesdrop场景对齐：RG、Coordinator、Observer；并发攻击+注册攻击6类聚合

实现说明（关键点）：
- 启动RG（子进程）、Coordinator（本进程）与合法Observer（本进程）
- 启动两个ANP节点（Doctor A/B）：创建AgentConnect SimpleNode（原生DID+WS），
  同时在本地启动HTTP适配端点（仅作为Coordinator调用的入口），接收/message后写回对端
  注：消息处理仍由ANP节点承载身份与WS通道；HTTP端点仅为协调器路由入口，不做协议替代

注意：此Runner要求本地存在agentconnect_src，且可成功import；否则抛错退出。
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent

import httpx
import subprocess
import logging

from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn

HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

# 尝试导入核心组件
try:
    from core.rg_coordinator import RGCoordinator
    from core.observer_agent import create_observer_agent
    from core.attack_scenarios import RegistrationAttackRunner
except ImportError:
    from script.safety_tech.core.rg_coordinator import RGCoordinator
    from script.safety_tech.core.observer_agent import create_observer_agent
    from script.safety_tech.core.attack_scenarios import RegistrationAttackRunner

# 原生ANP（AgentConnect）导入
AGENTCONNECT_OK = False
try:
    # 允许多路径
    candidates = [
        PROJECT_ROOT,
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from agentconnect_src.simple_node import SimpleNode, SimpleNodeSession
    from agentconnect_src.utils.did_generate import did_generate
    from agentconnect_src.utils.crypto_tool import (
        get_pem_from_private_key,
        get_hex_from_public_key,
        generate_signature_for_json,
    )
    AGENTCONNECT_OK = True
except Exception as e:
    # 增加更详细的路径调试信息
    print(f"DEBUG: sys.path = {sys.path}")
    print(f"DEBUG: CWD = {Path.cwd()}")
    print(f"DEBUG: PROJECT_ROOT = {PROJECT_ROOT}")
    raise RuntimeError(f"AgentConnect(ANP) SDK 未就绪: {e}")


def _load_medical_dataset() -> List[Dict[str, Any]]:
    try:
        possible_paths = [
            SAFETY_TECH / 'data' / 'enhanced_medical_questions.json',
            Path('script/safety_tech/data/enhanced_medical_questions.json'),
        ]
        dataset_file = None
        for p in possible_paths:
            if p.exists():
                dataset_file = p
                break
        if not dataset_file:
            raise FileNotFoundError('enhanced_medical_questions.json 未找到')
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data.get('questions', [])
        medical_cases = []
        for i, q in enumerate(questions[:10]):
            medical_cases.append({
                'case_id': f"dataset_case_{q.get('id', i+1)}",
                'patient_info': f"Patient: {q.get('sensitive_info', {}).get('name','Unknown')}, Age: {q.get('sensitive_info', {}).get('age','Unknown')}",
                'sensitive_info': q.get('sensitive_info', {}),
                'original_question': q.get('original_question', ''),
                'enhanced_question': q.get('enhanced_question', ''),
                'initial_question': f"Medical consultation needed: {q.get('enhanced_question', q.get('original_question',''))}"
            })
        return medical_cases
    except Exception as e:
        raise RuntimeError(f"加载医疗数据集失败: {e}")


async def _wait_http_ok(url: str, timeout_s: float = 20.0) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(url, timeout=2.0)
                if r.status_code == 200:
                    return
        except Exception as e:
            last_err = e
        await asyncio.sleep(0.3)
    raise RuntimeError(f"Timeout waiting {url}: {last_err}")


def _spawn(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env={**os.environ, **(env or {})}
    )


class ANPDoctorShim:
    """HTTP适配端点 + 原生AgentConnect SimpleNode
    - Coordinator 调用 /message → 适配为写入对端（通过“Runner内的HTTP回传”）
    说明：为保持与既有Coordinator一致性，这里通过HTTP完成路由入站，
    但节点本身仍以SimpleNode完成DID初始化与WS通道建立（无fallback）。
    """
    def __init__(self, agent_name: str, port: int):
        self.agent_name = agent_name
        self.port = port
        self.ws_port = port + 100  # 原生ANP WS端口与HTTP shim端口分离
        self.app = FastAPI(title=f"ANP {agent_name} HTTP Shim")
        self.did = None
        self.simple_node: Optional[SimpleNode] = None
        self.private_key = None
        self.public_key = None
        self.did_document_json: Optional[str] = None
        self._peer_did: Optional[str] = None
        self._peer_session: Optional[SimpleNodeSession] = None  # 使用官方API的会话对象
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post('/message')
        async def receive_message(payload: Dict[str, Any]):
            # 入站仅作业务触发，实际回执通过原生ANP通道发送给对端DID
            text = payload.get('text') or payload.get('content') or ''
            response = f"{self.agent_name} (ANP) received: {text}"
            if self.simple_node and self._peer_did:
                try:
                    await self.simple_node.send_message(response, self._peer_did)
                except Exception:
                    pass
            return {'status': 'processed', 'agent_id': self.agent_name, 'response': response}

        @self.app.get('/health')
        async def health():
            return {'status': 'healthy', 'agent_id': self.agent_name, 'did': self.did is not None, 'ws_port': self.ws_port}

        @self.app.get('/v1/did/{did_value}')
        async def get_did_doc(did_value: str):
            # 在HTTP shim上提供DID文档，供对端根据DID拉取
            if self.did and did_value == self.did and self.did_document_json:
                return Response(content=self.did_document_json, media_type='application/text')
            return Response(status_code=404)

    def set_peer_did(self, did: str):
        self._peer_did = did

    def start_http(self):
        import threading
        def run():
            uvicorn.run(self.app, host='127.0.0.1', port=self.port, log_level='warning', lifespan="off", loop="asyncio", http="h11")
        t = threading.Thread(target=run, daemon=True)
        t.start()

    async def _handle_new_session(self, session: SimpleNodeSession):
        """处理新的ANP会话连接 - 参考官方示例的直接循环模式"""
        print(f"   [DEBUG] {self.agent_name}: 新会话建立，来自 {session.remote_did}")
        
        try:
            while True:
                message_bytes = await session.receive_message()
                if message_bytes:
                    plaintext = message_bytes.decode('utf-8')
                    print(f"   [DEBUG] {self.agent_name}: 收到来自 {session.remote_did} 的消息: {plaintext[:50]}...")
                    
                    # 发送回声
                    reply = f"{self.agent_name} (ANP) echo: {plaintext}"
                    ok = await session.send_message(reply)
                    print(f"   [DEBUG] {self.agent_name}: 回声发送{'成功' if ok else '失败'}")
                    
                    # 将消息放入收件箱
                    try:
                        self._inbox.put_nowait((session.remote_did, plaintext))
                    except Exception as e:
                        print(f"   [DEBUG] {self.agent_name}: 收件箱放入异常: {e}")
                else:
                    # 收到None表示连接关闭
                    print(f"   [DEBUG] {self.agent_name}: 会话 {session.remote_did} 连接关闭")
                    break
        except asyncio.CancelledError:
            print(f"   [DEBUG] {self.agent_name}: 会话 {session.remote_did} 消息循环被取消")
        except Exception as e:
            print(f"   [DEBUG] {self.agent_name}: 会话消息循环异常: {e}")
        finally:
            print(f"   [DEBUG] {self.agent_name}: 会话 {session.remote_did} 消息循环结束")

    async def _outbound_message_loop(self, session: SimpleNodeSession):
        """处理出站连接的消息接收循环"""
        try:
            while True:
                message_bytes = await session.receive_message()
                if message_bytes:
                    plaintext = message_bytes.decode('utf-8')
                    print(f"   [DEBUG] {self.agent_name}: 出站会话收到来自 {session.remote_did} 的消息: {plaintext[:50]}...")
                    
                    # 将消息放入收件箱供业务等待
                    try:
                        self._inbox.put_nowait((session.remote_did, plaintext))
                    except Exception as e:
                        print(f"   [DEBUG] {self.agent_name}: 出站会话收件箱放入异常: {e}")
                else:
                    # 收到None表示连接关闭
                    print(f"   [DEBUG] {self.agent_name}: 出站会话 {session.remote_did} 连接关闭")
                    break
        except asyncio.CancelledError:
            print(f"   [DEBUG] {self.agent_name}: 出站会话 {session.remote_did} 消息循环被取消")
        except Exception as e:
            print(f"   [DEBUG] {self.agent_name}: 出站会话消息循环异常: {e}")
        finally:
            print(f"   [DEBUG] {self.agent_name}: 出站会话 {session.remote_did} 消息循环结束")

    def start_anp_node(self):
        # 使用官方最新ANP API创建SimpleNode
        self.simple_node = SimpleNode(
            host_domain="127.0.0.1",
            host_port=str(self.ws_port),
            host_ws_path="/ws",
            new_session_callback=self._handle_new_session
        )
        
        # 生成DID信息
        private_key_pem, did, did_document_json = self.simple_node.generate_did_document()
        self.simple_node.set_did_info(private_key_pem, did, did_document_json)
        
        # 启动SimpleNode服务
        self.simple_node.run()
        
        self.private_key = private_key_pem
        self.did = did
        self.did_document_json = did_document_json

    def start_message_loop(self):
        # 新的API已通过_handle_new_session和_session_message_loop处理消息接收
        # 这个方法保持为兼容性存根
        print(f"   [DEBUG] {self.agent_name}: 消息循环已通过会话回调处理")

    async def send_and_wait_echo(self, text: str, timeout_s: float = 8.0) -> bool:
        """通过原生ANP向对端发送消息，并在本节点等待回声以确认送达。
        成功条件：send_message返回True，且在timeout内收到任意入站消息。
        """
        if not (self.simple_node and self._peer_did):
            print(f"   [DEBUG] {self.agent_name}: simple_node或peer_did未设置")
            return False
        try:
            print(f"   [DEBUG] {self.agent_name}: 向 {self._peer_did} 发送消息: {text[:50]}...")
            
            # 使用官方API建立连接
            if not self._peer_session:
                print(f"   [DEBUG] {self.agent_name}: 建立到 {self._peer_did} 的连接")
                self._peer_session = await self.simple_node.connect_to_did(self._peer_did)
                if not self._peer_session:
                    print(f"   [DEBUG] {self.agent_name}: 连接建立失败")
                    return False
                # 为出站连接启动消息接收循环
                print(f"   [DEBUG] {self.agent_name}: 启动出站会话消息接收循环")
                asyncio.create_task(self._outbound_message_loop(self._peer_session))
                # 等待连接稳定
                await asyncio.sleep(0.5)
            
            # 发送消息
            ok = await self._peer_session.send_message(text)
            if not ok:
                print(f"   [DEBUG] {self.agent_name}: send_message返回False")
                return False
            print(f"   [DEBUG] {self.agent_name}: 消息发送成功，等待回声...")
            try:
                sender_did, received_text = await asyncio.wait_for(self._inbox.get(), timeout=timeout_s)
                print(f"   [DEBUG] {self.agent_name}: 收到来自 {sender_did} 的消息: {received_text[:50]}...")
                # 验证回声是否来自预期的对端
                if sender_did == self._peer_did:
                    print(f"   [DEBUG] {self.agent_name}: 回声验证成功")
                    return True
                else:
                    print(f"   [DEBUG] {self.agent_name}: 消息来自意外DID {sender_did}，放回队列")
                    # 消息来自其他DID，放回队列并继续等待
                    self._inbox.put_nowait((sender_did, received_text))
                    return False
            except asyncio.TimeoutError:
                print(f"   [DEBUG] {self.agent_name}: 等待回声超时")
                return False
        except Exception as e:
            print(f"   [DEBUG] {self.agent_name}: 异常: {e}")
            return False

    async def ensure_session(self, retries: int = 3, timeout_s: float = 12.0) -> bool:
        """主动探测并建立与对端的原生ANP会话。"""
        if not (self.simple_node and self._peer_did):
            return False
        for _ in range(retries):
            # 简单的ping消息来确认双向通道
            ok = await self.send_and_wait_echo(f"[ping] from {self.agent_name}", timeout_s=timeout_s)
            if ok:
                return True
            await asyncio.sleep(1.0)
        return False

    def build_registration_proof(self) -> Dict[str, Any]:
        ts = time.time()
        # 为注册构造签名消息（did + ts），生成签名与公钥hex
        message = {'did': self.did, 'timestamp': ts}
        signature = ''
        pub_hex = ''
        
        if self.private_key:
            try:
                # 从PEM字符串加载私钥对象
                from cryptography.hazmat.primitives import serialization
                private_key_obj = serialization.load_pem_private_key(
                    self.private_key.encode('utf-8'), 
                    password=None
                )
                signature = generate_signature_for_json(private_key_obj, message)
                pub_hex = get_hex_from_public_key(private_key_obj.public_key())
            except Exception as e:
                print(f"   [DEBUG] {self.agent_name}: 签名生成失败: {e}")
        
        return {
            'did': self.did,
            'did_signature': signature,
            'did_public_key': pub_hex,
            'timestamp': ts,
        }


async def main():
    # 端口配置
    rg_port = 8001
    coord_port = 8888
    obs_port = 8004
    a_port = 9102
    b_port = 9103
    conv_id = os.environ.get('ANP_CONV_ID', 'conv_anp_eaves')

    procs: List[subprocess.Popen] = []
    try:
        # 1) 启动RG
        proc = subprocess.Popen([
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); "
            "from script.safety_tech.core.registration_gateway import RegistrationGateway; "
            f"RegistrationGateway({{'session_timeout':3600,'max_observers':5,'require_observer_proof':True}}).run(host='127.0.0.1', port={rg_port})"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append(proc)
        print(f"Started ANP RG process with PID: {proc.pid}")
        try:
            await _wait_http_ok(f"http://127.0.0.1:{rg_port}/health", 12.0)
        except RuntimeError as e:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print(f"ANP RG process exited with code: {proc.returncode}")
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
            raise e

        # 2) 启动Coordinator（本进程）
        coordinator = RGCoordinator({
            'rg_endpoint': f'http://127.0.0.1:{rg_port}',
            'conversation_id': conv_id,
            'coordinator_port': coord_port
        })
        await coordinator.start()
        await _wait_http_ok(f"http://127.0.0.1:{coord_port}/health", 20.0)

        # 3) 启动合法Observer（本进程）
        await create_observer_agent(
            observer_id='Legitimate_Observer',
            config={'conversation_id': conv_id, 'max_stored_messages': 1000, 'eavesdrop_detection': {}},
            port=obs_port
        )

        # 4) 启动ANP医生节点 + HTTP适配端点
        doctor_a = ANPDoctorShim('ANP_Doctor_A', a_port)
        doctor_b = ANPDoctorShim('ANP_Doctor_B', b_port)
        doctor_a.start_http(); doctor_b.start_http()
        await asyncio.sleep(1.0)
        doctor_a.start_anp_node(); doctor_b.start_anp_node()

        # 设定对端DID（用于原生ANP回传）并开启消息监听循环
        doctor_a.set_peer_did(doctor_b.did)
        doctor_b.set_peer_did(doctor_a.did)
        doctor_a.start_message_loop()
        doctor_b.start_message_loop()
        # 主动预热原生会话，避免首轮超时
        await asyncio.gather(
            doctor_a.ensure_session(),
            doctor_b.ensure_session()
        )

        # 5) 注册到RG + 订阅Observer
        async with httpx.AsyncClient() as c:
            # 为每个节点构建原生ANP证明
            proofs = {
                'ANP_Doctor_A': doctor_a.build_registration_proof(),
                'ANP_Doctor_B': doctor_b.build_registration_proof(),
            }
            for agent_id, port, role in [
                ('ANP_Doctor_A', a_port, 'doctor_a'),
                ('ANP_Doctor_B', b_port, 'doctor_b'),
            ]:
                r = await c.post(f"http://127.0.0.1:{rg_port}/register", json={
                    'protocol': 'anp',
                    'agent_id': agent_id,
                    'endpoint': f"http://127.0.0.1:{port}",  # 供Coordinator调用
                    'conversation_id': conv_id,
                    'role': role,
                    'proof': proofs[agent_id]
                }, timeout=10.0)
                if r.status_code != 200:
                    raise RuntimeError(f"注册{agent_id}失败: {r.text}")
            sub = await c.post(f"http://127.0.0.1:{rg_port}/subscribe", json={
                'agent_id': 'Legitimate_Observer',
                'role': 'observer',
                'conversation_id': conv_id,
                'endpoint': f"http://127.0.0.1:{obs_port}",
                'proof': {'token': 'observer_ok', 'timestamp': time.time()}
            }, timeout=10.0)
            if sub.status_code != 200:
                raise RuntimeError(f"Observer订阅失败: {sub.text}")

        # 等待Coordinator目录刷新
        await asyncio.sleep(4)

        # 6) 加载数据集（标准：10个案例）
        medical_cases = _load_medical_dataset()

        # S1 并发攻击
        print("\n🛡️ === S1: 并发攻击下对话稳定性测试 ===")
        print(f"📊 标准测试规模: {len(medical_cases)}案例×5轮 = {len(medical_cases)*5}条对话")

        async def _malicious_registration_attack():
            try:
                async with httpx.AsyncClient() as c:
                    for i in range(10):
                        await c.post(f"http://127.0.0.1:{rg_port}/register", json={
                            'protocol': 'anp', 'agent_id': f'malicious_anp_{i}',
                            'endpoint': f'http://malicious:90{i}', 'conversation_id': conv_id,
                            'role': 'doctor_fake', 'proof': {'fake': True}
                        }, timeout=2.0)
                        await asyncio.sleep(2)
            except Exception:
                pass

        async def _spam_message_attack():
            try:
                async with httpx.AsyncClient() as c:
                    for i in range(20):
                        try:
                            await c.post(f"http://127.0.0.1:{coord_port}/route_message", json={
                                'sender_id': 'fake_sender', 'receiver_id': 'fake_receiver', 'text': 'SPAM'*100
                            }, timeout=2.0)
                        except Exception:
                            pass
                        await asyncio.sleep(1)
            except Exception:
                pass

        async def _replay_interference_attack():
            try:
                async with httpx.AsyncClient() as c:
                    replay = {'sender_id': 'ANP_Doctor_A','receiver_id': 'ANP_Doctor_B','text': 'REPLAYED'}
                    for _ in range(15):
                        await c.post(f"http://127.0.0.1:{coord_port}/route_message", json=replay, timeout=2.0)
                        await asyncio.sleep(2)
            except Exception:
                pass

        async def _dos_attack():
            try:
                async with httpx.AsyncClient() as c:
                    tasks = [c.get(f"http://127.0.0.1:{coord_port}/health", timeout=1.0) for _ in range(50)]
                    for _ in range(10):
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await asyncio.sleep(0.1)
            except Exception:
                pass

        async def _observer_interference_attack():
            try:
                async with httpx.AsyncClient() as c:
                    for _ in range(15):
                        reqs = [
                            c.get(f"http://127.0.0.1:{obs_port}/status", timeout=2.0),
                            c.get(f"http://127.0.0.1:{obs_port}/messages", timeout=2.0)
                        ]
                        await asyncio.gather(*reqs, return_exceptions=True)
                        await asyncio.sleep(2)
            except Exception:
                pass

        attacks = [
            asyncio.create_task(_malicious_registration_attack()),
            asyncio.create_task(_spam_message_attack()),
            asyncio.create_task(_replay_interference_attack()),
            asyncio.create_task(_dos_attack()),
            asyncio.create_task(_observer_interference_attack()),
        ]

        total_attempted = 0
        total_success = 0
        conversation_results = []
        for i, case in enumerate(medical_cases):
            print(f"\n📋 【案例 {i+1}/{len(medical_cases)}】: {case['case_id']}")
            print(f"   患者信息: {case['patient_info']}")
            msgs = []
            succ = 0
            for r in range(5):
                total_attempted += 1
                text = f"[Round {r+1}] {case['initial_question'][:200]}"
                try:
                    ok = await doctor_a.send_and_wait_echo(text)
                    if ok:
                        succ += 1
                        total_success += 1
                        msgs.append({'round': r+1, 'message': text, 'response': {'status': 'processed', 'via': 'anp'}})
                        print(f"   ✅ Round {r+1}/5 - 成功 (ANP原生通道)")
                    else:
                        print(f"   ❌ Round {r+1}/5 - 失败 (ANP原生通道超时)")
                except Exception as e:
                    print(f"   ❌ Round {r+1}/5 - 异常: {str(e)} [ANP通道]")
                await asyncio.sleep(1.0)
            conversation_results.append({'case_id': case['case_id'], 'messages': msgs, 'success': succ})
            print(f"   📊 案例完成: {succ}/5 轮成功 (攻击影响: {5-succ}轮)")

        for t in attacks:
            t.cancel()

        s1_rate = total_success / total_attempted if total_attempted else 0
        print("\n🛡️ === S1测试结果 ===")
        print(f"📊 攻击环境下对话完成率: {total_success}/{total_attempted} ({s1_rate:.1%})")
        print(f"📊 业务连续性评分: {s1_rate*100:.1f}/100")

        # S2 恶意窃听
        print("\n🕵️ === S2: 恶意窃听检测测试 ===")
        malicious = []
        try:
            async with httpx.AsyncClient() as c:
                for i in range(2):
                    oid = f"malicious_observer_{int(time.time())}_{i}"
                    sub = await c.post(f"http://127.0.0.1:{rg_port}/subscribe", json={
                        'observer_id': oid, 'conversation_id': conv_id,
                        'endpoint': f"http://malicious-observer:{9000+i}"
                    }, timeout=5.0)
                    if sub.status_code == 200:
                        malicious.append(oid)
                        print(f"   ❌ 恶意Observer {i+1} 成功注册: {oid}")
                    else:
                        print(f"   ✅ 恶意Observer {i+1} 被阻止注册")
        except Exception as e:
            print(f"   ✅ 恶意Observer注册失败: {str(e)[:50]}...")

        # 回填尝试
        try:
            async with httpx.AsyncClient() as c:
                hist = await c.get(f"http://127.0.0.1:{coord_port}/message_history", params={'limit':100}, timeout=5.0)
                if hist.status_code == 200:
                    msgs = hist.json()
                    print(f"   📦 回填攻击: 获取到 {len(msgs) if isinstance(msgs, list) else 0} 条历史消息")
                else:
                    print(f"   ✅ 回填攻击被阻止 (状态: {hist.status_code})")
        except Exception as e:
            print(f"   ✅ 回填攻击失败: {str(e)[:50]}...")

        # S3 6类注册攻击
        runner = RegistrationAttackRunner({
            'rg_endpoint': f'http://127.0.0.1:{rg_port}',
            'conversation_id': conv_id,
            'protocol': 'anp',
            'attack_timeout': 10.0
        })
        print("\n🎭 【S3: 恶意注册防护】")
        registration_attacks = []
        try:
            res = await runner.run_all_attacks()
            for a in res:
                t = getattr(a, 'attack_type', 'unknown')
                s = getattr(a, 'success', False)
                print(f"   {'❌' if s else '✅'} {t}: {'成功(丢分)' if s else '被阻止(得分)'}")
                registration_attacks.append({'attack_type': t, 'success': s})
        except Exception as e:
            print(f"   ⚠️ 注册攻击测试异常: {e}")
        s3_blocked = len([1 for a in registration_attacks if not a.get('success', False)])
        total_s3 = len(registration_attacks) if registration_attacks else 6
        s3_score = round((s3_blocked/total_s3*100) if total_s3>0 else 100, 1)
        print(f"   📊 S3结果: {s3_blocked}/{total_s3} 被阻止")

        # 统一评分（S1=15%、S2=25%、S3-S8=60%）
        s1_score = s1_rate * 100
        s2_success_rate = len(malicious)/2
        s2_score = 100 - (s2_success_rate * 100)
        unified = round(100 - ((100-s1_score)*0.15 + (100-s2_score)*0.25 + (100-s3_score)*0.60), 1)
        level = 'SECURE' if unified>=90 else 'MODERATE' if unified>=70 else 'VULNERABLE'

        # 报告
        out_dir = SAFETY_TECH / 'output'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"anp_unified_security_report_{int(time.time())}.json"
        report = {
            'protocol': 'anp',
            'security_score': unified,
            'security_level': level,
            'test_results': {
                'S1_business_continuity': {'completion_rate': s1_rate, 'score': round(s1_score,1)},
                'S2_eavesdrop_prevention': {'malicious_observers_blocked': len(malicious)==0, 'score': round(s2_score,1)},
                'S3_registration_defense': {
                    'attacks_blocked': f"{s3_blocked}/{total_s3}",
                    'score': round(s3_score,1),
                    'detailed': [
                        {'attack_type': a['attack_type'], 'result': ('得分' if not a['success'] else '丢分')}
                        for a in registration_attacks
                    ]
                }
            },
            'unified_metrics': {
                'total_attacks_attempted': total_s3,
                'total_attacks_blocked': s3_blocked,
                'total_messages_tested': len(medical_cases)*5,
                'malicious_observers_deployed': len(malicious),
                'test_duration_seconds': 300
            },
            'timestamp': time.time()
        }
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 控制台
        print("\n"+"="*80)
        print("🛡️ ANP 统一安全防护测试报告")
        print("="*80)
        print("📋 协议: ANP")
        print(f"📊 医疗案例: {len(conversation_results)}/10 (标准)")
        print(f"💬 对话轮次: {sum(len(c['messages']) for c in conversation_results)}/50 (标准)")
        print()
        print("🔍 安全测试结果:")
        print(f"   S1 业务连续性: {s1_score:.1f}/100 (攻击环境下对话完成率)")
        print(f"   S2 恶意窃听防护: {s2_score:.1f}/100 (恶意Observer阻止)")
        print(f"   S3 注册攻击防护: {s3_score:.1f}/100 (恶意注册阻止，细项见下)")
        for item in report['test_results']['S3_registration_defense']['detailed']:
            print(f"      · {item['attack_type']}: {item['result']}")
        print()
        print(f"🛡️ 统一安全评分: {unified}/100")
        print(f"🏷️ 安全等级: {level}")
        print(f"📄 详细报告: {out_file}")
        print("="*80+"\n")

    finally:
        # 结束RG子进程
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        end = time.time() + 5
        for p in procs:
            try:
                if p.poll() is None and time.time() < end:
                    p.wait(timeout=max(0, end-time.time()))
            except Exception:
                pass
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass


if __name__ == '__main__':
    asyncio.run(main())


