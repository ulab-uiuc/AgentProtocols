"""
Conflict detection and resolution for concurrent MAPF.
"""

import asyncio
import time
from typing import Dict, List, Tuple, Optional
from .concurrent_types import *
import logging

class ConflictManager:
    """Manages time-space reservations and conflict detection"""
    
    def __init__(self):
        self.reservations: Dict[Coord, List[ReservationRecord]] = {}
        self.pending_moves: Dict[str, ConcurrentMoveCmd] = {}
        self.logger = logging.getLogger("ConflictManager")
        # 🔧 CRITICAL: Limit lookahead to prevent long reservations
        self.MAX_LOOKAHEAD_MS = 500  # Maximum 500ms reservation window
        
    def check_move_conflicts(self, move: ConcurrentMoveCmd) -> MoveResponse:
        """Check if a move request conflicts with existing reservations"""
        current_time = int(time.time() * 1000)
        
        # 🔧 CRITICAL: Limit eta_ms to prevent long horizon reservations
        # If eta_ms is too large (indicating absolute timestamp), cap it
        effective_eta = min(move.eta_ms, self.MAX_LOOKAHEAD_MS)
        
        # Calculate time window for this move
        start_time = current_time + effective_eta - move.time_window_ms
        end_time = current_time + effective_eta + move.time_window_ms
        
        # Check for conflicts in target cell
        cell = move.new_pos
        conflicting_agents = []
        
        if cell in self.reservations:
            for reservation in self.reservations[cell]:
                # Check time overlap
                if (start_time < reservation.end_time_ms and 
                    end_time > reservation.start_time_ms):
                    conflicting_agents.append(reservation.agent_id)
        
        if conflicting_agents:
            return MoveResponse(
                agent_id=move.agent_id,
                move_id=move.move_id,
                status=MoveStatus.CONFLICT,
                reason=f"Cell {cell} occupied during time window",
                conflicting_agents=conflicting_agents,
                suggested_eta_ms=self._suggest_alternative_time(move, conflicting_agents)
            )
        else:
            # No conflict - create reservation
            reservation = ReservationRecord(
                agent_id=move.agent_id,
                cell=cell,
                start_time_ms=start_time,
                end_time_ms=end_time,
                move_id=move.move_id,
                priority=move.priority
            )
            
            if cell not in self.reservations:
                self.reservations[cell] = []
            self.reservations[cell].append(reservation)
            
            return MoveResponse(
                agent_id=move.agent_id,
                move_id=move.move_id,
                status=MoveStatus.OK,
                reason="Move approved",
                reservation_id=move.move_id
            )
    
    def _suggest_alternative_time(self, move: ConcurrentMoveCmd, conflicting_agents: List[int]) -> int:
        """Suggest an alternative eta that avoids conflicts"""
        cell = move.new_pos
        current_time = int(time.time() * 1000)
        
        # Find the latest end time of conflicting reservations
        latest_end = current_time
        if cell in self.reservations:
            for reservation in self.reservations[cell]:
                if reservation.agent_id in conflicting_agents:
                    latest_end = max(latest_end, reservation.end_time_ms)
        
        # Suggest eta that starts after conflicts end
        suggested_eta = latest_end - current_time + move.time_window_ms + 50  # 50ms buffer
        return max(suggested_eta, move.eta_ms + 100)  # At least 100ms later than original
    
    def release_reservation(self, move_id: str, agent_id: int):
        """Release a reservation when move is completed or cancelled"""
        for cell, reservations in self.reservations.items():
            self.reservations[cell] = [
                r for r in reservations 
                if not (r.move_id == move_id and r.agent_id == agent_id)
            ]
    
    def cleanup_expired_reservations(self):
        """Remove reservations that have expired"""
        current_time = int(time.time() * 1000)
        
        for cell in list(self.reservations.keys()):
            self.reservations[cell] = [
                r for r in self.reservations[cell]
                if r.end_time_ms > current_time - 5000  # Keep 5s buffer for late cleanup
            ]
            
            if not self.reservations[cell]:
                del self.reservations[cell]
    
    def get_conflict_summary(self) -> Dict:
        """Get summary of current conflicts for monitoring"""
        current_time = int(time.time() * 1000)
        active_reservations = 0
        
        for reservations in self.reservations.values():
            active_reservations += len([
                r for r in reservations 
                if r.end_time_ms > current_time
            ])
        
        return {
            "active_cells": len(self.reservations),
            "active_reservations": active_reservations,
            "pending_moves": len(self.pending_moves)
        }