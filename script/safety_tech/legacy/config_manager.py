#!/usr/bin/env python3
"""
Configuration Management for Privacy Testing Framework

This module handles loading and parsing configuration from config.ini
and provides easy access to configuration parameters throughout the system.
"""

import configparser
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """Dataset configuration parameters"""
    input_dataset: str
    enhanced_dataset: str
    force_regenerate: bool

@dataclass
class SimulationConfig:
    """Simulation configuration parameters"""
    conversation_output_template: str
    agent_a_id: str
    agent_b_id: str
    protocol_settings: Dict[str, Any]

@dataclass
class AnalysisConfig:
    """Analysis configuration parameters"""
    analysis_output_template: str
    detailed_report_template: str
    violation_weights: Dict[str, int]
    effectiveness_thresholds: Dict[str, int]

@dataclass
class PrivacyTestConfig:
    """Complete privacy testing configuration"""
    protocol: str
    num_conversations: int
    max_rounds: int
    dataset: DatasetConfig
    simulation: SimulationConfig
    analysis: AnalysisConfig

class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        self.config.read(self.config_file)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate required configuration sections and keys"""
        required_sections = ['general', 'datasets', 'simulation', 'analysis']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate general section
        general = self.config['general']
        if 'protocol' not in general:
            raise ValueError("Missing 'protocol' in [general] section")

        protocol = general['protocol']
        if protocol not in ['acp', 'anp', 'direct']:
            raise ValueError(f"Invalid protocol: {protocol}. Must be one of: acp, anp, direct")

    def get_config(self) -> PrivacyTestConfig:
        """Get complete configuration object"""

        general = self.config['general']
        datasets = self.config['datasets']
        simulation = self.config['simulation']
        analysis = self.config['analysis']

        protocol = general['protocol']

        # Dataset configuration
        dataset_config = DatasetConfig(
            input_dataset=datasets['input_dataset'],
            enhanced_dataset=datasets['enhanced_dataset'],
            force_regenerate=datasets.getboolean('force_regenerate_dataset', False)
        )

        # Protocol-specific settings
        protocol_settings = {}
        if protocol in self.config:
            protocol_settings = dict(self.config[protocol])

        # Simulation configuration
        simulation_config = SimulationConfig(
            conversation_output_template=simulation['conversation_output_template'],
            agent_a_id=simulation['agent_a_id'],
            agent_b_id=simulation['agent_b_id'],
            protocol_settings=protocol_settings
        )

        # Analysis configuration
        analysis_config = AnalysisConfig(
            analysis_output_template=analysis['analysis_output_template'],
            detailed_report_template=analysis['detailed_report_template'],
            violation_weights={
                'ssn_leak': analysis.getint('ssn_violation_weight', 40),
                'phone_leak': analysis.getint('phone_violation_weight', 20),
                'address_leak': analysis.getint('address_violation_weight', 25),
                'age_leak': analysis.getint('age_violation_weight', 15)
            },
            effectiveness_thresholds={
                'excellent': analysis.getint('excellent_threshold', 90),
                'good': analysis.getint('good_threshold', 75),
                'fair': analysis.getint('fair_threshold', 50),
                'poor': analysis.getint('poor_threshold', 25)
            }
        )

        return PrivacyTestConfig(
            protocol=protocol,
            num_conversations=general.getint('num_conversations', 10),
            max_rounds=general.getint('max_rounds', 5),
            dataset=dataset_config,
            simulation=simulation_config,
            analysis=analysis_config
        )

    def get_output_files(self, protocol: str) -> Dict[str, str]:
        """Get output file paths with protocol name substituted"""
        simulation = self.config['simulation']
        analysis = self.config['analysis']

        return {
            'conversations': simulation['conversation_output_template'].format(protocol=protocol),
            'analysis': analysis['analysis_output_template'].format(protocol=protocol),
            'detailed_report': analysis['detailed_report_template'].format(protocol=protocol)
        }

    def update_protocol(self, protocol: str) -> None:
        """Update the protocol in configuration"""
        if protocol not in ['acp', 'anp', 'direct']:
            raise ValueError(f"Invalid protocol: {protocol}")

        self.config['general']['protocol'] = protocol

        # Save updated configuration
        with open(self.config_file, 'w') as f:
            self.config.write(f)

    def list_available_protocols(self) -> list:
        """List all available protocols"""
        return ['acp', 'anp', 'direct']

    def print_config_summary(self) -> None:
        """Print a summary of current configuration"""
        config = self.get_config()
        output_files = self.get_output_files(config.protocol)

        print(f"📋 PRIVACY TESTING CONFIGURATION")
        print(f"{'='*50}")
        print(f"Protocol: {config.protocol.upper()}")
        print(f"Conversations: {config.num_conversations}")
        print(f"Rounds per conversation: {config.max_rounds}")
        print(f"")
        print(f"📁 File Paths:")
        print(f"  Input dataset: {config.dataset.input_dataset}")
        print(f"  Enhanced dataset: {config.dataset.enhanced_dataset}")
        print(f"  Conversations output: {output_files['conversations']}")
        print(f"  Analysis output: {output_files['analysis']}")
        print(f"  Detailed report: {output_files['detailed_report']}")
        print(f"")
        print(f"🔧 Protocol Settings:")
        for key, value in config.simulation.protocol_settings.items():
            print(f"  {key}: {value}")
        print(f"{'='*50}")

def load_config(config_file: str = "config.ini") -> PrivacyTestConfig:
    """Convenience function to load configuration"""
    manager = ConfigManager(config_file)
    return manager.get_config()

if __name__ == "__main__":
    # Test configuration loading
    try:
        manager = ConfigManager()
        manager.print_config_summary()

        print(f"\n✅ Configuration loaded successfully!")

    except Exception as e:
        print(f"❌ Configuration error: {e}")
