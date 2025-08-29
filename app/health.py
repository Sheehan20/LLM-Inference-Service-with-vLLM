from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, Optional
import structlog
from app.config import Settings
from app.metrics import set_health_status, HEALTH_STATUS


logger = structlog.get_logger()


class HealthChecker:
    """Comprehensive health monitoring for the inference service."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.checks = {
            "gpu_memory": self._check_gpu_memory,
            "engine_ready": self._check_engine_ready,
            "system_resources": self._check_system_resources,
            "response_time": self._check_response_time,
        }
        self.last_check_time = 0
        self.check_interval = 30  # seconds
        self.response_times = []
        self.max_response_time = 5.0  # seconds
    
    async def run_health_checks(self, engine_manager=None) -> Dict[str, Any]:
        """Run all health checks and return status."""
        current_time = time.time()
        
        # Rate limit health checks
        if current_time - self.last_check_time < self.check_interval:
            return {"status": "cached", "message": "Using cached health status"}
        
        self.last_check_time = current_time
        results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                if check_name == "engine_ready" and engine_manager:
                    result = await check_func(engine_manager)
                else:
                    result = await check_func()
                
                results[check_name] = result
                set_health_status(check_name, result["healthy"])
                
            except Exception as e:
                logger.exception(f"Health check {check_name} failed", error=str(e))
                results[check_name] = {"healthy": False, "error": str(e)}
                set_health_status(check_name, False)
        
        # Overall health status
        overall_healthy = all(check["healthy"] for check in results.values())
        set_health_status("overall", overall_healthy)
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": results,
            "timestamp": current_time
        }
    
    async def _check_gpu_memory(self) -> Dict[str, Any]:
        """Check GPU memory usage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return {"healthy": False, "message": "No GPU devices found"}
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                usage_percent = (mem_info.used / mem_info.total) * 100
                
                if usage_percent > 95:
                    return {
                        "healthy": False, 
                        "message": f"GPU {i} memory usage too high: {usage_percent:.1f}%",
                        "gpu_memory_usage": usage_percent
                    }
            
            pynvml.nvmlShutdown()
            return {"healthy": True, "message": "GPU memory usage normal"}
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_engine_ready(self, engine_manager) -> Dict[str, Any]:
        """Check if the inference engine is ready."""
        if not engine_manager or not engine_manager._engine:
            return {"healthy": False, "message": "Engine not initialized"}
        
        try:
            # Try a quick health check with the engine
            # This could be extended to do a quick test inference
            return {"healthy": True, "message": "Engine ready"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return {
                    "healthy": False, 
                    "message": f"High CPU usage: {cpu_percent}%",
                    "cpu_usage": cpu_percent
                }
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {
                    "healthy": False,
                    "message": f"High memory usage: {memory.percent}%",
                    "memory_usage": memory.percent
                }
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return {
                    "healthy": False,
                    "message": f"Low disk space: {100-disk.percent}% free",
                    "disk_usage": disk.percent
                }
            
            return {
                "healthy": True, 
                "message": "System resources normal",
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_response_time(self) -> Dict[str, Any]:
        """Check average response time."""
        if not self.response_times:
            return {"healthy": True, "message": "No response time data yet"}
        
        # Keep only recent response times (last 100)
        self.response_times = self.response_times[-100:]
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        if avg_response_time > self.max_response_time:
            return {
                "healthy": False,
                "message": f"Average response time too high: {avg_response_time:.2f}s",
                "avg_response_time": avg_response_time
            }
        
        return {
            "healthy": True,
            "message": f"Response time normal: {avg_response_time:.2f}s",
            "avg_response_time": avg_response_time
        }
    
    def record_response_time(self, response_time: float):
        """Record a response time for monitoring."""
        self.response_times.append(response_time)


# Global health checker instance
health_checker: Optional[HealthChecker] = None


def get_health_checker(settings: Settings = None) -> HealthChecker:
    """Get or create the global health checker instance."""
    global health_checker
    if health_checker is None and settings:
        health_checker = HealthChecker(settings)
    return health_checker