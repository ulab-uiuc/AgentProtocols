"""Logging utilities for MAPF project.
Provides:
    • Colored console output.
    • Session-based log directory under script/async_mapf/log/<timestamp>/
    • Per-agent log files: agent<ID>-<protocol>.log
    • Network log file: network.log
    • Helper methods to record LLM interaction / agent actions / network events.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# ========== Colored output ==========
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    BLACK = "\033[30m"; RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"; BRIGHT_RED = "\033[91m"; BRIGHT_GREEN = "\033[92m"; BRIGHT_YELLOW = "\033[93m"; BRIGHT_BLUE = "\033[94m"; BRIGHT_MAGENTA = "\033[95m"; BRIGHT_CYAN = "\033[96m"; BRIGHT_WHITE = "\033[97m"

class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        'DEBUG': Colors.BRIGHT_BLACK,
        'INFO': Colors.BRIGHT_GREEN,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BRIGHT_MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        text = super().format(record)
        # Color level name
        color = self.LEVEL_COLORS.get(record.levelname, "")
        text = text.replace(record.levelname, f"{color}{record.levelname}{Colors.RESET}", 1)
        return text

# ========== Session Log Manager ==========
class SessionLogManager:
    """Manage per-run log directory and loggers"""

    def __init__(self, base_dir: str = "script/async_mapf/log", debug_config: Dict[str, Any] = None) -> None:
        self.session_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._agent_loggers: Dict[str, logging.Logger] = {}
        self._network_logger: logging.Logger | None = None
        
        # Debug configuration
        self.debug_config = debug_config or {}
        self.verbose = self.debug_config.get("verbose", False)
        self.log_level = self.debug_config.get("log_level", "INFO")
        self.log_llm_interactions = self.debug_config.get("log_llm_interactions", True)
        self.log_agent_actions = self.debug_config.get("log_agent_actions", True)
        self.log_network_events = self.debug_config.get("log_network_events", True)

    # ------------- helpers -------------
    def _build_file_logger(self, name: str, file_path: Path) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # File handler
        fh = logging.FileHandler(file_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        # Console handler (colored) - respect log_level config
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        console_level = getattr(logging, self.log_level, logging.INFO)
        ch.setLevel(console_level)
        logger.addHandler(ch)
        return logger

    # ------------- public -------------
    def get_agent_logger(self, agent_id: int, protocol: str) -> logging.Logger:
        key = f"agent{agent_id}-{protocol}"
        if key not in self._agent_loggers:
            file_path = self.session_dir / f"{key}.log"
            self._agent_loggers[key] = self._build_file_logger(key, file_path)
        return self._agent_loggers[key]

    def get_network_logger(self) -> logging.Logger:
        if self._network_logger is None:
            file_path = self.session_dir / "network.log"
            self._network_logger = self._build_file_logger("network", file_path)
        return self._network_logger

    # -------- recording helpers --------
    def log_llm_interaction(self, agent_id: int, protocol: str, input_data: Any, output_data: Any):
        if not self.log_llm_interactions:
            return
        logger = self.get_agent_logger(agent_id, protocol)
        if input_data is not None:
            logger.info("🔍 LLM INPUT:")
            if self.verbose:
                logger.info(str(input_data))
            else:
                logger.info(f"[INPUT DATA - {type(input_data).__name__}]")
        if output_data is not None:
            logger.info("📤 LLM OUTPUT:")
            if self.verbose:
                logger.info(str(output_data))
            else:
                logger.info(f"[OUTPUT DATA - {type(output_data).__name__}]")
        logger.info("="*40)

    def log_agent_action(self, agent_id: int, protocol: str, action_type: str, data: Any):
        if not self.log_agent_actions:
            return
        logger = self.get_agent_logger(agent_id, protocol)
        if self.verbose:
            logger.info(f"🎯 {action_type}: {data}")
        else:
            logger.info(f"🎯 {action_type}")

    def log_network_event(self, event_type: str, data: Any):
        if not self.log_network_events:
            return
        logger = self.get_network_logger()
        if self.verbose:
            logger.info(f"🌐 {event_type}: {data}")
        else:
            logger.info(f"🌐 {event_type}")

# ========== module-level helpers ==========
_global_mgr: SessionLogManager | None = None

def get_log_manager(debug_config: Dict[str, Any] = None) -> SessionLogManager:
    global _global_mgr
    if _global_mgr is None:
        _global_mgr = SessionLogManager(debug_config=debug_config)
    return _global_mgr

def setup_colored_logging():
    """Setup colored logging with protection against duplicate handlers."""
    root = logging.getLogger()
    
    # 🔧 CRITICAL FIX: 检查是否已经有StreamHandler，避免重复清理
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    
    if has_stream_handler:
        # 已经有console handler，不要再清理
        return
    
    # 只在没有StreamHandler时才清理并添加新的
    root.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch.setLevel(logging.INFO)
    root.addHandler(ch)
    root.setLevel(logging.INFO)

def log_agent_action(agent_id: int, protocol: str, action_type: str, data: Any):
    get_log_manager().log_agent_action(agent_id, protocol, action_type, data)

def log_network_event(event_type: str, data: Any):
    get_log_manager().log_network_event(event_type, data)
