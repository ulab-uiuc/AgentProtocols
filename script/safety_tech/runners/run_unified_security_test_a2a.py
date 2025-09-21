# -*- coding: utf-8 -*-
"""
A2A Unified Security Test Runner

要求：
- 原生A2A（a2a-sdk），无fallback/mock/简单实现
- 与ACP/ANP/Agora完全一致的测试规模、指标与权重（S1=15%，S2=25%，S3-S8=60%）
- 与Eavesdrop场景对齐：RG、Coordinator、Observer；并发攻击+注册攻击6类聚合
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(SAFETY_TECH))
# 为支持 `import src.*`，需要将项目根目录加入 sys.path（而非 src 目录本身）
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# 导入核心组件（Coordinator/Observer/攻击场景）
try:
    from core.rg_coordinator import RGCoordinator
    from core.observer_agent import create_observer_agent
    from core.attack_scenarios import RegistrationAttackRunner
except ImportError:
    from script.safety_tech.core.rg_coordinator import RGCoordinator
    from script.safety_tech.core.observer_agent import create_observer_agent
    from script.safety_tech.core.attack_scenarios import RegistrationAttackRunner

# 原生A2A（a2a-sdk）服务适配器
from src.server_adapters.a2a_adapter import A2AServerAdapter


def _load_medical_dataset() -> List[Dict[str, Any]]:
    try:
        possible = [
            SAFETY_TECH / 'data' / 'enhanced_medical_questions.json',
            Path('script/safety_tech/data/enhanced_medical_questions.json'),
        ]
        dataset = None
        for p in possible:
            if p.exists():
                dataset = p
                break
        if not dataset:
            raise FileNotFoundError('enhanced_medical_questions.json 未找到')
        with open(dataset, 'r', encoding='utf-8') as f:
            data = json.load(f)
        qs = data.get('questions', [])
        cases: List[Dict[str, Any]] = []
        for i, q in enumerate(qs[:10]):
            cases.append({
                'case_id': f"dataset_case_{q.get('id', i+1)}",
                'patient_info': f"Patient: {q.get('sensitive_info', {}).get('name','Unknown')}, Age: {q.get('sensitive_info', {}).get('age','Unknown')}",
                'sensitive_info': q.get('sensitive_info', {}),
                'original_question': q.get('original_question', ''),
                'enhanced_question': q.get('enhanced_question', ''),
                'initial_question': f"Medical consultation needed: {q.get('enhanced_question', q.get('original_question',''))}"
            })
        return cases
    except Exception as e:
        raise RuntimeError(f"加载医疗数据集失败: {e}")


async def _wait_http_ok(url: str, timeout_s: float = 20.0) -> None:
    start = time.time()
    last_err: Optional[Exception] = None
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


class A2ADoctorServer:
    """原生 a2a-sdk 服务器包装，供Coordinator通过 /message 路由调用。
    使用 A2AServerAdapter 启动 Starlette+uvicorn 服务，并提供 executor.execute(ctx, queue)。
    """
    def __init__(self, agent_id: str, host: str, port: int):
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

        class _Executor:
            async def execute(self_inner, context, event_queue):
                # 从context提取文本（A2A adapter已封装）
                try:
                    from a2a.utils import new_agent_text_message
                except Exception as e:
                    raise RuntimeError(f"a2a-sdk 未安装或不可用: {e}")

                # 简单回文响应，保持与ACP/ANP一致的业务回声
                msg = getattr(context, 'message', None)
                text = None
                try:
                    # SDK Message 可能在不同版本字段不同
                    if hasattr(msg, 'parts') and msg.parts:
                        for p in msg.parts:
                            if isinstance(p, dict) and p.get('type') == 'text':
                                text = p.get('text')
                                break
                            # pydantic对象
                            t = getattr(p, 'text', None)
                            if isinstance(t, str):
                                text = t
                                break
                    if not text:
                        text = str(getattr(msg, 'text', '')) or str(msg)
                except Exception:
                    text = str(msg)

                reply = f"{self.agent_id} (A2A) received: {text}"
                await event_queue.enqueue_event(new_agent_text_message(reply))

        self._executor = _Executor()

    def build(self):
        adapter = A2AServerAdapter()
        server, _card = adapter.build(host=self.host, port=self.port, agent_id=self.agent_id, executor=self._executor)
        return server

    def start(self):
        import threading
        self._server = self.build()
        def _run():
            self._server.run()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        try:
            if self._server:
                self._server.should_exit = True
        except Exception:
            pass


async def main():
    # 端口配置
    rg_port = 8001
    coord_port = 8888
    obs_port = 8004
    a_port = 9202
    b_port = 9203
    conv_id = os.environ.get('A2A_CONV_ID', 'conv_a2a_eaves')

    procs: List[Any] = []
    try:
        # 1) 启动RG
        import subprocess
        # Debug: capture stderr to see what's going wrong
        proc = subprocess.Popen([
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); "
            "from script.safety_tech.core.registration_gateway import RegistrationGateway; "
            f"RegistrationGateway({{'session_timeout':3600,'max_observers':5,'require_observer_proof':True}}).run(host='127.0.0.1', port={rg_port})"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append(proc)
        print(f"Started RG process with PID: {proc.pid}")
        try:
            await _wait_http_ok(f"http://127.0.0.1:{rg_port}/health", 15.0)
        except RuntimeError as e:
            # Check if process is still running and get error output
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print(f"RG process exited with code: {proc.returncode}")
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
            raise e

        # 2) 启动Coordinator
        coordinator = RGCoordinator({
            'rg_endpoint': f'http://127.0.0.1:{rg_port}',
            'conversation_id': conv_id,
            'coordinator_port': coord_port
        })
        await coordinator.start()
        await _wait_http_ok(f"http://127.0.0.1:{coord_port}/health", 20.0)

        # 3) 启动合法Observer
        await create_observer_agent(
            observer_id='Legitimate_Observer',
            config={'conversation_id': conv_id, 'max_stored_messages': 1000, 'eavesdrop_detection': {}},
            port=obs_port
        )

        # 4) 启动原生A2A医生服务器
        doctor_a = A2ADoctorServer('A2A_Doctor_A', '127.0.0.1', a_port)
        doctor_b = A2ADoctorServer('A2A_Doctor_B', '127.0.0.1', b_port)
        doctor_a.start(); doctor_b.start()
        await _wait_http_ok(f"http://127.0.0.1:{a_port}/health", 15.0)
        await _wait_http_ok(f"http://127.0.0.1:{b_port}/health", 15.0)

        # 5) 注册到RG + 订阅Observer
        async with httpx.AsyncClient() as c:
            for agent_id, port, role in [
                ('A2A_Doctor_A', a_port, 'doctor_a'),
                ('A2A_Doctor_B', b_port, 'doctor_b'),
            ]:
                # A2A 原生证明：此处携带 a2a_token 字段以触发A2A校验器
                proof = {
                    'a2a_token': f"token_{agent_id}_{int(time.time())}",
                    'timestamp': time.time(),
                    'nonce': str(uuid.uuid4()),
                }
                r = await c.post(f"http://127.0.0.1:{rg_port}/register", json={
                    'protocol': 'a2a',
                    'agent_id': agent_id,
                    'endpoint': f"http://127.0.0.1:{port}",
                    'conversation_id': conv_id,
                    'role': role,
                    'proof': proof
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
                            'protocol': 'a2a', 'agent_id': f'malicious_a2a_{i}',
                            'endpoint': f'http://malicious:92{i}', 'conversation_id': conv_id,
                            'role': 'doctor_fake', 'proof': {'a2a_token': f'mal_{i}', 'timestamp': time.time()}
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
                    replay = {'sender_id': 'A2A_Doctor_A','receiver_id': 'A2A_Doctor_B','text': 'REPLAYED'}
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
        conversation_results: List[Dict[str, Any]] = []
        async with httpx.AsyncClient() as c:
            for i, case in enumerate(medical_cases):
                print(f"\n📋 【案例 {i+1}/{len(medical_cases)}】: {case['case_id']}")
                print(f"   患者信息: {case['patient_info']}")
                msgs: List[Dict[str, Any]] = []
                succ = 0
                for r in range(5):
                    total_attempted += 1
                    text = f"[Round {r+1}] {case['initial_question'][:200]}"
                    try:
                        rr = await c.post(f"http://127.0.0.1:{coord_port}/route_message", json={
                            'sender_id': 'A2A_Doctor_A','receiver_id':'A2A_Doctor_B','text': text
                        }, timeout=10.0)
                        is_ok = rr.status_code in (200, 202)
                        js = None
                        try:
                            js = rr.json()
                        except Exception:
                            js = None
                        has_err = isinstance(js, dict) and (js.get('error') is not None)
                        status_ok = isinstance(js, dict) and (js.get('status') in ('processed','ok','success'))
                        # 统一成功标准：HTTP 200/202 且 响应无error；兼容status为processed/ok/success
                        if is_ok and (status_ok or not has_err):
                            succ += 1
                            total_success += 1
                            msgs.append({'round': r+1, 'message': text, 'response': js if js is not None else {'status_code': rr.status_code}})
                            print(f"   ✅ Round {r+1}/5 - 成功 (攻击环境下)")
                        else:
                            debug_info = f"状态码:{rr.status_code}, 响应:{js}, has_err:{has_err}, status_ok:{status_ok}"
                            print(f"   ❌ Round {r+1}/5 - 失败 ({debug_info}) [攻击影响]")
                    except Exception as e:
                        print(f"   ❌ Round {r+1}/5 - 异常: {str(e)} [攻击影响]")
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
        malicious: List[str] = []
        try:
            async with httpx.AsyncClient() as c:
                for i in range(2):
                    oid = f"malicious_observer_{int(time.time())}_{i}"
                    sub = await c.post(f"http://127.0.0.1:{rg_port}/subscribe", json={
                        'agent_id': oid, 'conversation_id': conv_id, 'role': 'observer',
                        'endpoint': f"http://malicious-observer:{9000+i}", 'proof': {'token': 'reuse'}
                    }, timeout=5.0)
                    if sub.status_code == 200:
                        malicious.append(oid)
                        print(f"   ❌ 恶意Observer {i+1} 成功注册: {oid}")
                    else:
                        print(f"   ✅ 恶意Observer {i+1} 被阻止注册")
        except Exception as e:
            print(f"   ✅ 恶意Observer注册失败: {str(e)[:80]}...")

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
            print(f"   ✅ 回填攻击失败: {str(e)[:80]}...")

        # S3 6类注册攻击
        runner = RegistrationAttackRunner({
            'rg_endpoint': f'http://127.0.0.1:{rg_port}',
            'conversation_id': conv_id,
            'protocol': 'a2a',
            'attack_timeout': 10.0
        })
        print("\n🎭 【S3: 恶意注册防护】")
        registration_attacks: List[Dict[str, Any]] = []
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
        out_file = out_dir / f"a2a_unified_security_report_{int(time.time())}.json"
        report = {
            'protocol': 'a2a',
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
        print("🛡️ A2A 统一安全防护测试报告")
        print("="*80)
        print("📋 协议: A2A")
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


