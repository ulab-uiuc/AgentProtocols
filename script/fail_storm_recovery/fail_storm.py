#!/usr/bin/env python3
"""
Fail-Storm Injection Script

This script implements the fault injection mechanism for the Fail-Storm Recovery scenario.
It randomly kills a specified percentage of Agent processes at a predetermined time.

Key Features:
1. Time-delayed execution (default: 60 seconds)
2. Configurable kill percentage (default: 30%)
3. Random victim selection
4. SIGKILL signal for immediate termination
5. Detailed logging of killed processes
"""

import os
import signal
import time
import random
import json
import sys
import argparse
from typing import List, Dict, Any
from pathlib import Path


class FailStormInjector:
    """
    Fault injection system for simulating sudden agent failures.
    
    This class manages the process of identifying, selecting, and terminating
    agent processes to simulate realistic failure scenarios in distributed systems.
    """
    
    def __init__(self, agent_pids: List[int], kill_fraction: float = 0.3, delay: float = 60.0):
        """
        Initialize the FailStorm injector.
        
        Parameters
        ----------
        agent_pids : List[int]
            List of Agent process IDs to potentially kill
        kill_fraction : float
            Fraction of agents to kill (0.0 to 1.0)
        delay : float
            Delay in seconds before executing the kill
        """
        self.agent_pids = agent_pids.copy()
        self.kill_fraction = max(0.0, min(1.0, kill_fraction))  # Clamp to [0, 1]
        self.delay = delay
        
        # Execution state
        self.victims: List[int] = []
        self.killed_pids: List[int] = []
        self.failed_kills: List[Dict[str, Any]] = []
        self.execution_time: float = 0.0
        
        # Validation
        if not self.agent_pids:
            raise ValueError("No agent PIDs provided")
        
        print(f"[FailStorm] Initialized with {len(self.agent_pids)} agents, "
              f"kill_fraction={self.kill_fraction:.1%}, delay={self.delay}s")

    def select_victims(self) -> List[int]:
        """
        Randomly select victim processes to kill.
        
        Returns
        -------
        List[int]
            List of PIDs selected for termination
        """
        if not self.agent_pids:
            return []
        
        # Calculate number of victims
        num_victims = max(1, int(len(self.agent_pids) * self.kill_fraction))
        num_victims = min(num_victims, len(self.agent_pids))  # Don't exceed available
        
        # Randomly select victims
        self.victims = random.sample(self.agent_pids, num_victims)
        
        print(f"[FailStorm] Selected {len(self.victims)} victims out of {len(self.agent_pids)} agents:")
        for i, pid in enumerate(self.victims, 1):
            print(f"  {i}. PID {pid}")
        
        return self.victims

    def execute_kills(self) -> Dict[str, Any]:
        """
        Execute the kill operations on selected victim processes.
        
        Returns
        -------
        Dict[str, Any]
            Summary of kill execution results
        """
        if not self.victims:
            raise RuntimeError("No victims selected. Call select_victims() first.")
        
        print(f"[FailStorm] Executing kills on {len(self.victims)} processes...")
        self.execution_time = time.time()
        
        for pid in self.victims:
            try:
                # Check if process exists before killing
                if not self._process_exists(pid):
                    self.failed_kills.append({
                        "pid": pid,
                        "error": "Process not found",
                        "timestamp": time.time()
                    })
                    print(f"  ❌ PID {pid}: Process not found")
                    continue
                
                # Send SIGKILL signal
                os.kill(pid, signal.SIGKILL)
                self.killed_pids.append(pid)
                print(f"  ✅ PID {pid}: SIGKILL sent")
                
                # Brief pause between kills to avoid overwhelming the system
                time.sleep(0.1)
                
            except ProcessLookupError:
                self.failed_kills.append({
                    "pid": pid,
                    "error": "Process already terminated",
                    "timestamp": time.time()
                })
                print(f"  ⚠️  PID {pid}: Already terminated")
                
            except PermissionError:
                self.failed_kills.append({
                    "pid": pid,
                    "error": "Permission denied",
                    "timestamp": time.time()
                })
                print(f"  ❌ PID {pid}: Permission denied")
                
            except Exception as e:
                self.failed_kills.append({
                    "pid": pid,
                    "error": str(e),
                    "timestamp": time.time()
                })
                print(f"  ❌ PID {pid}: {e}")
        
        # Generate execution summary
        summary = {
            "execution_time": self.execution_time,
            "total_targets": len(self.victims),
            "successful_kills": len(self.killed_pids),
            "failed_kills": len(self.failed_kills),
            "success_rate": len(self.killed_pids) / len(self.victims) if self.victims else 0,
            "killed_pids": self.killed_pids.copy(),
            "failed_attempts": self.failed_kills.copy()
        }
        
        print(f"[FailStorm] Execution completed: "
              f"{summary['successful_kills']}/{summary['total_targets']} kills successful "
              f"({summary['success_rate']:.1%} success rate)")
        
        return summary

    def _process_exists(self, pid: int) -> bool:
        """Check if a process with given PID exists."""
        try:
            # Send signal 0 to check if process exists without affecting it
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't have permission (still counts as existing)
            return True

    def run_with_delay(self) -> Dict[str, Any]:
        """
        Execute the complete fail-storm sequence with delay.
        
        Returns
        -------
        Dict[str, Any]
            Complete execution results including timing information
        """
        start_time = time.time()
        
        print(f"[FailStorm] Starting fail-storm sequence...")
        print(f"[FailStorm] Waiting {self.delay} seconds before execution...")
        
        # Wait for the specified delay
        time.sleep(self.delay)
        
        # Select victims
        self.select_victims()
        
        # Execute kills
        kill_summary = self.execute_kills()
        
        # Calculate total timing
        total_time = time.time() - start_time
        
        # Generate complete results
        complete_results = {
            "metadata": {
                "start_time": start_time,
                "delay_seconds": self.delay,
                "total_runtime": total_time,
                "kill_fraction": self.kill_fraction,
                "random_seed": None  # Could add seed support for reproducibility
            },
            "target_selection": {
                "total_agents": len(self.agent_pids),
                "selected_victims": len(self.victims),
                "victim_pids": self.victims.copy()
            },
            "execution_results": kill_summary
        }
        
        return complete_results

    def save_results(self, filepath: str, results: Dict[str, Any]) -> None:
        """
        Save fail-storm results to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the results JSON file
        results : Dict[str, Any]
            Results dictionary to save
        """
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"[FailStorm] Results saved to: {output_path}")
            
        except Exception as e:
            print(f"[FailStorm] Failed to save results: {e}")


def load_pids_from_file(filepath: str) -> List[int]:
    """
    Load agent PIDs from a file.
    
    Parameters
    ----------
    filepath : str
        Path to file containing PIDs (one per line or JSON format)
        
    Returns
    -------
    List[int]
        List of PIDs loaded from file
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            
        # Try JSON format first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [int(pid) for pid in data]
            elif isinstance(data, dict) and 'pids' in data:
                return [int(pid) for pid in data['pids']]
        except json.JSONDecodeError:
            pass
        
        # Try line-by-line format
        lines = content.split('\n')
        pids = []
        for line in lines:
            line = line.strip()
            if line and line.isdigit():
                pids.append(int(line))
        
        return pids
        
    except Exception as e:
        print(f"Error loading PIDs from {filepath}: {e}")
        return []


def main():
    """Main entry point for the fail-storm script."""
    parser = argparse.ArgumentParser(
        description="Fail-Storm Injection Script - Randomly kill agent processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Kill 30% of agents after 60 seconds
  python fail_storm.py --pids 1234,5678,9012
  
  # Kill 50% of agents after 30 seconds
  python fail_storm.py --pids-file agent_pids.txt --fraction 0.5 --delay 30
  
  # Immediate execution (for testing)
  python fail_storm.py --pids 1234,5678 --delay 0
        """
    )
    
    # PID input options
    pid_group = parser.add_mutually_exclusive_group(required=True)
    pid_group.add_argument(
        '--pids',
        type=str,
        help='Comma-separated list of agent PIDs to potentially kill'
    )
    pid_group.add_argument(
        '--pids-file',
        type=str,
        help='File containing agent PIDs (one per line or JSON format)'
    )
    
    # Execution parameters
    parser.add_argument(
        '--fraction', '-f',
        type=float,
        default=0.3,
        help='Fraction of agents to kill (0.0 to 1.0, default: 0.3)'
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=60.0,
        help='Delay in seconds before killing agents (default: 60.0)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='fail_storm_results.json',
        help='Output file for results (default: fail_storm_results.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Load PIDs
    if args.pids:
        try:
            agent_pids = [int(pid.strip()) for pid in args.pids.split(',')]
        except ValueError:
            print("Error: Invalid PID format. Use comma-separated integers.")
            sys.exit(1)
    else:
        agent_pids = load_pids_from_file(args.pids_file)
        if not agent_pids:
            print(f"Error: No valid PIDs found in {args.pids_file}")
            sys.exit(1)
    
    # Validate parameters
    if not (0.0 <= args.fraction <= 1.0):
        print("Error: Fraction must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.delay < 0:
        print("Error: Delay cannot be negative")
        sys.exit(1)
    
    # Execute fail-storm
    try:
        injector = FailStormInjector(
            agent_pids=agent_pids,
            kill_fraction=args.fraction,
            delay=args.delay
        )
        
        results = injector.run_with_delay()
        
        # Save results
        injector.save_results(args.output, results)
        
        # Print summary
        print("\n" + "="*60)
        print("FAIL-STORM EXECUTION SUMMARY")
        print("="*60)
        print(f"Total agents: {len(agent_pids)}")
        print(f"Kill fraction: {args.fraction:.1%}")
        print(f"Victims selected: {len(injector.victims)}")
        print(f"Successful kills: {len(injector.killed_pids)}")
        print(f"Failed kills: {len(injector.failed_kills)}")
        print(f"Success rate: {results['execution_results']['success_rate']:.1%}")
        print(f"Results saved to: {args.output}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()