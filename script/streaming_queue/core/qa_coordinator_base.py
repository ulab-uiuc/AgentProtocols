# -*- coding: utf-8 -*-
"""
QA Coordinator Base:
- 负责：配置/加载题目/动态调度/结果汇总与落盘
- 不关心：具体通信协议
- 留出抽象方法 send_to_worker() 由具体协议实现（如 A2A、HTTP、gRPC 等）

使用方式：
    from qa_coordinator_base import QACoordinatorBase

    class MyCoordinator(QACoordinatorBase):
        async def send_to_worker(self, worker_id: str, question: str) -> dict:
            ...
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class QACoordinatorBase(ABC):
    """QA Coordinator 抽象基类：调度逻辑完整，通信交给子类实现。"""

    def __init__(self, config: Optional[dict] = None, output=None):
        self.config = config or {}
        qa_cfg = self.config.get("qa", {}) or {}
        coordinator_cfg = qa_cfg.get("coordinator", {}) or {}

        self.batch_size: int = int(coordinator_cfg.get("batch_size", 50))
        self.first_50: bool = bool(coordinator_cfg.get("first_50", True))
        self.data_path: str = coordinator_cfg.get("data_file", "data/top1000_simplified.jsonl")
        self.result_file: str = coordinator_cfg.get("result_file", "data/qa_results.json")

        self.coordinator_id: str = coordinator_cfg.get("coordinator_id", "Coordinator-1")
        self.worker_ids: List[str] = []
        self.agent_network: Any = None  # 仅作为存放“网络/路由器/客户端”的容器，含义由子类定义
        self.output = output

    # --------------- 基础开关 ---------------
    def set_network(self, network: Any, worker_ids: List[str]) -> None:
        """设置网络/路由器与可用 worker 列表。network 的具体类型由子类解释。"""
        self.agent_network = network
        self.worker_ids = list(worker_ids)

    # --------------- I/O 工具 ---------------
    def _o(self, level: str, msg: str) -> None:
        if not self.output:
            return
        try:
            fn = getattr(self.output, level, None)
            if callable(fn):
                fn(msg)
        except Exception:
            pass

    # --------------- 题库读取 ---------------
    async def load_questions(self) -> List[Dict[str, str]]:
        """从 JSONL 读取题目，返回 [{'id':..., 'question':...}, ...]"""
        questions: List[Dict[str, str]] = []
        
        # 如果是相对路径，相对于 streaming_queue 目录
        if not Path(self.data_path).is_absolute():
            # 找到 streaming_queue 目录
            current_file = Path(__file__).resolve()
            streaming_queue_dir = current_file.parent.parent  # 从 core 目录回到 streaming_queue
            p = streaming_queue_dir / self.data_path
        else:
            p = Path(self.data_path)

        if not p.exists():
            self._o("error", f"Question file does not exist: {p}")
            return questions

        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    q = item.get("q", "")
                    qid = item.get("id", str(len(questions) + 1))
                    if q:
                        questions.append({"id": qid, "question": q})
                    if self.first_50 and len(questions) >= 50:
                        break
                except json.JSONDecodeError as e:
                    self._o("error", f"JSON parsing failed: {line[:50]}... Error: {e}")

        self._o("system", f"Loaded {len(questions)} questions")
        return questions

    # --------------- 抽象通信 ---------------
    @abstractmethod
    async def send_to_worker(self, worker_id: str, question: str) -> Dict[str, Any]:
        """
        抽象：把 question 发给指定 worker，并返回一个标准化结果字典：
            {
                "answer": <str 或 None>,
                "raw": <协议原始返回，可选>,
            }
        子类必须实现（例如：A2A、HTTP、gRPC 等）。
        """
        raise NotImplementedError

    # --------------- 调度：动态负载 ---------------
    async def dispatch_questions_dynamically(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not self.worker_ids:
            self._o("warning", "No workers configured.")
            self._o("warning", f"Debug info - agent_network: {self.agent_network is not None}, worker_ids: {self.worker_ids}")
            return []
        self._o("info", f"Starting dynamic dispatch: {len(questions)} questions, {len(self.worker_ids)} workers")
        self._o("info", f"Worker IDs: {self.worker_ids}")
        self._o("info", f"Network available: {self.agent_network is not None}")

        q_queue: asyncio.Queue = asyncio.Queue()
        for q in questions:
            await q_queue.put(q)
        r_queue: asyncio.Queue = asyncio.Queue()

        workers = [
            asyncio.create_task(self._worker_loop(worker_id, q_queue, r_queue))
            for worker_id in self.worker_ids
        ]
        collector = asyncio.create_task(self._collect_results(r_queue, len(questions)))

        results = await collector
        await asyncio.gather(*workers, return_exceptions=True)
        return results

    async def _worker_loop(self, worker_id: str, q_queue: asyncio.Queue, r_queue: asyncio.Queue) -> None:
        processed = 0
        while True:
            try:
                item = q_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            processed += 1
            qid = item["id"]
            qtext = item["question"]
            self._o("progress", f"{worker_id} processing {qid}: {qtext[:50]}...")

            start = time.time()
            try:
                resp = await self.send_to_worker(worker_id, qtext)
                answer = (resp or {}).get("answer") or "No answer received"
                status = "success" if answer and answer != "No answer received" else "failed"
                err = None
            except Exception as e:
                answer, status, err = None, "failed", str(e)
                resp = None
            end = time.time()

            await r_queue.put({
                "question_id": qid,
                "question": qtext,
                "worker": worker_id,
                "answer": answer,
                "response_time": (end - start) if status == "success" else None,
                "timestamp": time.time(),
                "status": status,
                "error": err,
                "raw": resp,
            })
            q_queue.task_done()

        self._o("system", f"{worker_id} completed, processed {processed} questions")

    async def _collect_results(self, r_queue: asyncio.Queue, total: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        got = 0
        while got < total:
            item = await r_queue.get()
            out.append(item)
            got += 1
            if got % 5 == 0 or got == total:
                self._o("system", f"Collected {got}/{total} results")
            r_queue.task_done()
        self._o("success", f"Result collection completed, collected {got} results")
        return out

    # --------------- 一键执行并落盘 ---------------
    async def dispatch_round(self) -> str:
        questions = await self.load_questions()
        if not questions:
            return "Error: No questions loaded"

        t0 = time.time()
        results = await self.dispatch_questions_dynamically(questions)
        t1 = time.time()

        await self.save_results(results)

        succ = sum(1 for r in results if r["status"] == "success")
        fail = sum(1 for r in results if r["status"] == "failed")
        summary = (
            f"Dispatch round completed in {t1 - t0:.2f} seconds\n"
            f"Total processed: {len(results)}\n"
            f"Successfully processed: {succ}\n"
            f"Failed: {fail}\n"
            f"Workers used: {len(self.worker_ids)}\n"
            f"Results saved to file"
        )
        return summary

    async def save_results(self, results: List[Dict[str, Any]]) -> None:
        try:
            # 如果是相对路径，相对于 streaming_queue 目录
            if not Path(self.result_file).is_absolute():
                current_file = Path(__file__).resolve()
                streaming_queue_dir = current_file.parent.parent  # 从 core 目录回到 streaming_queue
                p = streaming_queue_dir / self.result_file
            else:
                p = Path(self.result_file)
            p.parent.mkdir(parents=True, exist_ok=True)

            times = [r["response_time"] for r in results if r.get("response_time")]
            avg_rt = (sum(times) / len(times)) if times else 0.0
            payload = {
                "metadata": {
                    "total_questions": len(results),
                    "successful_questions": sum(1 for r in results if r["status"] == "success"),
                    "failed_questions": sum(1 for r in results if r["status"] == "failed"),
                    "average_response_time": avg_rt,
                    "timestamp": time.time(),
                    "network_type": self.__class__.__name__,
                },
                "results": results,
            }
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self._o("success", f"Results saved to: {p}")
        except Exception as e:
            self._o("error", f"Failed to save results: {e}")

    # --------------- 状态查询 ---------------
    async def get_status(self) -> str:
        net = "Connected" if self.agent_network else "Not connected"
        return (
            "QA Coordinator Status:\n"
            f"Configuration: batch_size={self.batch_size}, first_50={self.first_50}\n"
            f"Data path: {self.data_path}\n"
            f"Network status: {net}\n"
            f"Worker count: {len(self.worker_ids)}\n"
            f"Available commands: dispatch, status"
        )
