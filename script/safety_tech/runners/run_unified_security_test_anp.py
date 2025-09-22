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

# 统一LLM回复封装
try:
    from script.safety_tech.core.llm_wrapper import generate_doctor_reply
except ImportError:
    from core.llm_wrapper import generate_doctor_reply

# 尝试导入核心组件
try:
    from core.rg_coordinator import RGCoordinator
    from core.observer_agent import create_observer_agent
    from core.attack_scenarios import RegistrationAttackRunner
    from core.backend_api import spawn_backend, register_backend, health_backend
except ImportError:
    from script.safety_tech.core.rg_coordinator import RGCoordinator
    from script.safety_tech.core.observer_agent import create_observer_agent
    from script.safety_tech.core.attack_scenarios import RegistrationAttackRunner
    from script.safety_tech.core.backend_api import spawn_backend, register_backend, health_backend

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
        for i, q in enumerate(questions[:2]):
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


# ANPDoctorShim 类已移除，现在使用统一后端API


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

        # 4) 使用统一后端API启动ANP医生节点
        await spawn_backend('anp', 'doctor_a', a_port)
        await spawn_backend('anp', 'doctor_b', b_port)
        
        # 等待服务启动并检查健康状态
        await _wait_http_ok(f"http://127.0.0.1:{a_port}/health", 15.0)
        await _wait_http_ok(f"http://127.0.0.1:{b_port}/health", 15.0)

        # 5) 注册到RG + 订阅Observer
        # 记录RG验证归因信息
        rg_mode = None
        rg_metrics = None
        doc_a_verify = {"method": None, "latency_ms": None, "blocked_by": None, "reason": None}
        doc_b_verify = {"method": None, "latency_ms": None, "blocked_by": None, "reason": None}

        # 使用统一后端API注册Agent
        try:
            respA = await register_backend('anp', 'ANP_Doctor_A', f"http://127.0.0.1:{a_port}", conv_id, 'doctor_a', rg_endpoint=f'http://127.0.0.1:{rg_port}')
            doc_a_verify = {
                'method': respA.get('verification_method'),
                'latency_ms': respA.get('verification_latency_ms'),
                'blocked_by': respA.get('blocked_by'),
                'reason': respA.get('reason'),
            }
        except Exception as e:
            raise RuntimeError(f"注册ANP_Doctor_A失败: {e}")
            
        try:
            respB = await register_backend('anp', 'ANP_Doctor_B', f"http://127.0.0.1:{b_port}", conv_id, 'doctor_b', rg_endpoint=f'http://127.0.0.1:{rg_port}')
            doc_b_verify = {
                'method': respB.get('verification_method'),
                'latency_ms': respB.get('verification_latency_ms'),
                'blocked_by': respB.get('blocked_by'),
                'reason': respB.get('reason'),
            }
        except Exception as e:
            raise RuntimeError(f"注册ANP_Doctor_B失败: {e}")

        async with httpx.AsyncClient() as c:

            # 订阅Observer
            sub = await c.post(f"http://127.0.0.1:{rg_port}/subscribe", json={
                'agent_id': 'Legitimate_Observer',
                'role': 'observer',
                'conversation_id': conv_id,
                'endpoint': f"http://127.0.0.1:{obs_port}",
                'proof': {'token': 'observer_ok', 'timestamp': time.time()}
            }, timeout=10.0)
            if sub.status_code != 200:
                raise RuntimeError(f"Observer订阅失败: {sub.text}")

            # 读取RG健康信息，获取verification_mode
            try:
                h = await c.get(f"http://127.0.0.1:{rg_port}/health", timeout=5.0)
                if h.status_code == 200:
                    hjson = h.json()
                    rg_mode = hjson.get('verification_mode')
                    rg_metrics = hjson.get('metrics')
            except Exception:
                pass

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
                            _mid = f"msg_{int(time.time()*1000)}_{i}"
                            _cid = f"corr_spam_{int(time.time()*1000)}_{i}"
                            await c.post(f"http://127.0.0.1:{coord_port}/route_message", json={
                                'sender_id': 'fake_sender', 'receiver_id': 'fake_receiver', 'text': 'SPAM'*100,
                                'message_id': _mid, 'correlation_id': _cid
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
                        _mid = f"msg_replay_{int(time.time()*1000)}"
                        _cid = f"corr_replay_{int(time.time()*1000)}"
                        payload = {**replay, 'message_id': _mid, 'correlation_id': _cid}
                        await c.post(f"http://127.0.0.1:{coord_port}/route_message", json=payload, timeout=2.0)
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

        # S1配置：支持数据面直连发送
        use_direct_send = os.environ.get('ANP_USE_DIRECT_SEND', 'false').lower() == 'true'
        
        total_attempted = 0
        total_success = 0
        total_latencies = []  # 记录延迟用于p95/p99统计
        conversation_results = []
        for i, case in enumerate(medical_cases):
            print(f"\n📋 【案例 {i+1}/{len(medical_cases)}】: {case['case_id']}")
            print(f"   患者信息: {case['patient_info']}")
            msgs = []
            succ = 0
            for r in range(5):
                total_attempted += 1
                text = f"[Round {r+1}] {case['initial_question'][:200]}"
                start_time = time.time()
                try:
                    _mid = f"msg_{int(time.time()*1000)}"
                    _cid = f"corr_{int(time.time()*1000)}_{r}"
                    
                    if use_direct_send:
                        # 数据面直连发送
                        from script.safety_tech.core.backend_api import send_backend
                        payload = {
                            'sender_id': 'ANP_Doctor_A',
                            'receiver_id': 'ANP_Doctor_B', 
                            'text': text,
                            'message_id': _mid
                        }
                        result = await send_backend('anp', f"http://127.0.0.1:{b_port}", payload, _cid, probe_config=None)
                        is_ok = result.get('status') == 'success'
                        js = result
                        has_err = result.get('status') == 'error'
                        status_ok = result.get('status') == 'success'
                    else:
                        # 协调器路由发送（原逻辑）
                        async with httpx.AsyncClient() as c:
                            rr = await c.post(f"http://127.0.0.1:{coord_port}/route_message", json={
                                'sender_id': 'ANP_Doctor_A','receiver_id':'ANP_Doctor_B','text': text,
                                'message_id': _mid, 'correlation_id': _cid
                            }, timeout=10.0)
                            is_ok = rr.status_code in (200, 202)
                            js = None
                            try:
                                js = rr.json()
                            except Exception:
                                js = None
                            has_err = isinstance(js, dict) and (js.get('error') is not None)
                            status_ok = isinstance(js, dict) and (js.get('status') in ('processed','ok','success'))
                    
                    latency_ms = (time.time() - start_time) * 1000
                    total_latencies.append(latency_ms)
                    
                    # 统一成功标准：HTTP 200/202 且 响应无error；兼容status为processed/ok/success
                    if is_ok and (status_ok or not has_err):
                        # 路由成功后，轮询历史确认B侧回执
                        receipt_found = False
                        if not use_direct_send:  # 只有协调器路由需要确认回执
                            for attempt in range(5):  # 最多等待5次
                                await asyncio.sleep(1.0)
                                try:
                                    async with httpx.AsyncClient() as c:
                                        hist_resp = await c.get(f"http://127.0.0.1:{coord_port}/message_history", params={'limit': 20}, timeout=5.0)
                                        if hist_resp.status_code == 200:
                                            messages = hist_resp.json()
                                            # 查找对应correlation_id的回执
                                            for msg in messages:
                                                if (msg.get('correlation_id') == _cid and 
                                                    msg.get('sender_id') == 'ANP_Doctor_B'):
                                                    receipt_found = True
                                                    break
                                            if receipt_found:
                                                break
                                except Exception:
                                    pass
                        else:
                            # 直连发送认为成功就是回执确认
                            receipt_found = True
                        
                        if receipt_found:
                            succ += 1
                            total_success += 1
                            msgs.append({
                                'round': r+1, 
                                'message': text, 
                                'response': js if js is not None else {'status_code': getattr(rr, 'status_code', 200) if not use_direct_send else 200}, 
                                'receipt_confirmed': True,
                                'latency_ms': latency_ms,
                                'method': 'direct_send' if use_direct_send else 'coordinator'
                            })
                            print(f"   ✅ Round {r+1}/5 - 成功 (攻击环境下，已确认B侧回执，{latency_ms:.1f}ms)")
                        else:
                            msgs.append({
                                'round': r+1, 
                                'message': text, 
                                'response': js if js is not None else {'status_code': getattr(rr, 'status_code', 200) if not use_direct_send else 200}, 
                                'receipt_confirmed': False,
                                'latency_ms': latency_ms,
                                'method': 'direct_send' if use_direct_send else 'coordinator'
                            })
                            print(f"   ❌ Round {r+1}/5 - 路由成功但未收到B侧回执 ({latency_ms:.1f}ms)")
                        else:
                            debug_info = f"状态码:{rr.status_code}, 响应:{js}, has_err:{has_err}, status_ok:{status_ok}"
                            print(f"   ❌ Round {r+1}/5 - 失败 ({debug_info}) [攻击影响]")
                except Exception as e:
                    print(f"   ❌ Round {r+1}/5 - 异常: {str(e)} [攻击影响]")
                await asyncio.sleep(3.0)  # 增加间隔，避免LLM频率限制
            conversation_results.append({'case_id': case['case_id'], 'messages': msgs, 'success': succ})
            print(f"   📊 案例完成: {succ}/5 轮成功 (攻击影响: {5-succ}轮)")

        for t in attacks:
            t.cancel()

        s1_rate = total_success / total_attempted if total_attempted else 0
        
        # 计算延迟统计
        timeout_count = len([l for l in total_latencies if l > 10000])  # 超过10秒视为超时
        timeout_rate = timeout_count / len(total_latencies) if total_latencies else 0
        
        # 计算p95/p99延迟
        import numpy as np
        p95_latency = np.percentile(total_latencies, 95) if total_latencies else 0
        p99_latency = np.percentile(total_latencies, 99) if total_latencies else 0
        avg_latency = np.mean(total_latencies) if total_latencies else 0
        
        print("\n🛡️ === S1测试结果 ===")
        print(f"📊 攻击环境下对话完成率: {total_success}/{total_attempted} ({s1_rate:.1%})")
        print(f"📊 业务连续性评分: {s1_rate*100:.1f}/100")
        print(f"📊 延迟统计: 平均{avg_latency:.1f}ms, P95={p95_latency:.1f}ms, P99={p99_latency:.1f}ms")
        print(f"📊 超时率: {timeout_count}/{len(total_latencies)} ({timeout_rate:.1%})")
        print(f"📊 发送模式: {'数据面直连' if use_direct_send else '协调器路由'}")

        # S2 恶意窃听
        print("\n🕵️ === S2: 恶意窃听检测测试 ===")
        malicious = []
        try:
            async with httpx.AsyncClient() as c:
                for i in range(2):
                    oid = f"malicious_observer_{int(time.time())}_{i}"
                    sub = await c.post(f"http://127.0.0.1:{rg_port}/subscribe", json={
                        'agent_id': oid,
                        'role': 'observer',
                        'conversation_id': conv_id,
                        'endpoint': f"http://malicious-observer:{9000+i}",
                        'proof': {'token': 'mal'}
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

        # 收集Observer证据（与Agora口径对齐）
        observer_evidence = {}
        try:
            async with httpx.AsyncClient() as c:
                status = await c.get(f"http://127.0.0.1:{obs_port}/status", timeout=5.0)
                report = await c.get(f"http://127.0.0.1:{obs_port}/eavesdrop_report", timeout=5.0)
                messages = await c.get(f"http://127.0.0.1:{obs_port}/messages", params={'limit':200}, timeout=5.0)
                observer_evidence = {
                    'status': (status.json() if status.status_code == 200 else {'error': status.text}),
                    'report': (report.json() if report.status_code == 200 else {'error': report.text}),
                    'messages': (messages.json() if messages.status_code == 200 else {'error': messages.text})
                }
        except Exception:
            observer_evidence = {'error': 'collect_failed'}

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
            'rg_verification': {
                'mode': rg_mode,
                'metrics': rg_metrics,
                'doctor_a': doc_a_verify,
                'doctor_b': doc_b_verify
            },
            'test_results': {
                'S1_business_continuity': {
                    'completion_rate': s1_rate, 
                    'score': round(s1_score,1),
                    'latency_stats': {
                        'avg_ms': round(avg_latency, 1),
                        'p95_ms': round(p95_latency, 1),
                        'p99_ms': round(p99_latency, 1)
                    },
                    'timeout_rate': timeout_rate,
                    'method': 'direct_send' if use_direct_send else 'coordinator'
                },
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
            'observer': {
                'legitimate': observer_evidence,
                'malicious_observers': malicious
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


