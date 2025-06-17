"""
Mono-deployment service module for running the FastAPI server and Temporal worker together.

This script implements the first part of the task to create a mono-deployment mechanism.
It enables starting both the API server and Temporal worker process from a single command,
with configuration options for scaling activities and worker processes.
"""

import os
import signal
import asyncio
import time
import argparse
import uvicorn
import subprocess
import logging
from multiprocessing import Process
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("truss.service")

# Global process registry
processes = {}

def parse_args():
    """Parse command line arguments for the service."""
    parser = argparse.ArgumentParser(description="Start Truss services in mono-deployment mode")
    
    # General settings
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server to")
    parser.add_argument("--api-only", action="store_true", help="Start only the API server")
    parser.add_argument("--worker-only", action="store_true", help="Start only the worker")
    
    # Worker settings
    parser.add_argument("--worker-count", type=int, default=1, help="Number of worker processes to start")
    parser.add_argument("--activity-scale", type=str, action='append', 
                        help="Scale specific activities: --activity-scale activity_name:count")
    
    # Environment settings
    parser.add_argument("--temporal-server", default=os.getenv("TEMPORAL_SERVER", "localhost:7233"),
                        help="Temporal server address")
    parser.add_argument("--postgres-url", default=os.getenv("POSTGRES_URL"),
                        help="PostgreSQL connection URL")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379"),
                        help="Redis connection URL")
    
    return parser.parse_args()

def run_api_server(host: str, port: int, env: Dict[str, str]):
    """Run the FastAPI server using uvicorn."""
    # Set environment variables for the API server
    for key, value in env.items():
        os.environ[key] = value
    
    # Import here to ensure environment variables are set before imports
    from .api import app
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=3)

def run_worker(env: Dict[str, str], activity_scales: Dict[str, int] = None):
    """Run the Temporal worker with custom activity scales."""
    # Set environment variables for the worker
    for key, value in env.items():
        os.environ[key] = value
    
    # Import here to ensure environment variables are set before imports
    from .run_worker import main as start_worker
    
    logger.info("Starting Temporal worker")
    
    # Convert activity scales to the format expected by the worker
    activities_config = {}
    if activity_scales:
        for activity_name, scale in activity_scales.items():
            activities_config[activity_name] = {"max_concurrent": scale}
    
    # Start the worker
    asyncio.run(start_worker(activities_config=activities_config))

def start_worker_subprocess(env: Dict[str, str], activity_scales: Dict[str, int] = None):
    """Start a worker subprocess with the given environment and activity scales."""
    worker_env = os.environ.copy()
    worker_env.update(env)
    
    cmd = ["python", "-m", "truss.run_worker"]
    
    # Add activity scale arguments if provided
    if activity_scales:
        for activity, scale in activity_scales.items():
            cmd.extend(["--activity-scale", f"{activity}:{scale}"])
    
    logger.info(f"Starting worker subprocess with command: {cmd}")
    return subprocess.Popen(cmd, env=worker_env)

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down processes."""
    logger.info(f"Received signal {sig}, shutting down...")
    for name, process in processes.items():
        logger.info(f"Terminating {name} process...")
        if isinstance(process, Process):
            process.terminate()
        else:  # subprocess.Popen
            process.send_signal(signal.SIGTERM)
    
    # Wait for processes to terminate
    for name, process in processes.items():
        if isinstance(process, Process):
            process.join(timeout=5)
        else:  # subprocess.Popen
            process.wait(timeout=5)
        logger.info(f"{name} process terminated")
    
    logger.info("All processes terminated, exiting")
    exit(0)

def parse_activity_scales(scale_args: List[str]) -> Dict[str, int]:
    """Parse activity scale arguments into a dictionary."""
    if not scale_args:
        return {}
    
    result = {}
    for arg in scale_args:
        try:
            activity, count = arg.split(":")
            result[activity.strip()] = int(count.strip())
        except ValueError:
            logger.warning(f"Invalid activity scale format: {arg}. Expected format: activity_name:count")
    
    return result

def main():
    """Main entry point for the mono-deployment service."""
    args = parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Prepare environment variables
    env = {
        "TEMPORAL_SERVER": args.temporal_server,
        "POSTGRES_URL": args.postgres_url,
        "REDIS_URL": args.redis_url,
    }
    
    # Filter out None values
    env = {k: v for k, v in env.items() if v is not None}
    
    # Parse activity scales
    activity_scales = parse_activity_scales(args.activity_scale)
    
    # Start processes based on arguments
    if not args.worker_only:
        logger.info("Starting API server process")
        api_process = Process(
            target=run_api_server,
            args=(args.host, args.port, env),
            daemon=True
        )
        api_process.start()
        processes["api"] = api_process
    
    if not args.api_only:
        logger.info(f"Starting {args.worker_count} worker processes")
        for i in range(args.worker_count):
            worker_process = start_worker_subprocess(env, activity_scales)
            processes[f"worker-{i}"] = worker_process
    
    # Wait for any process to terminate
    try:
        while True:
            for name, process in list(processes.items()):
                if isinstance(process, Process) and not process.is_alive():
                    logger.error(f"{name} process terminated unexpectedly")
                    signal_handler(None, None)
                elif isinstance(process, subprocess.Popen) and process.poll() is not None:
                    logger.error(f"{name} process terminated unexpectedly with code {process.returncode}")
                    signal_handler(None, None)
            
            # Sleep to avoid high CPU usage
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 
