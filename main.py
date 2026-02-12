import os
import asyncio
import signal
import json
import time
import numpy as np
from contextlib import asynccontextmanager
import contextlib
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from lib.px4_multirotor_edit import PX4Multirotor
from rotorpy.vehicles.px4_sihsim_quadx_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj

hover_trajectory = HoverTraj(x0=np.array([0, 0, 5]))

class SimServer:
    def __init__(self):
        # 1. Initialize as None so we don't block startup
        self.vehicle = None 
        self.controller = SE3Control(quad_params)

        self.target_fps = 150.0
        self.dt_target = 1.0 / self.target_fps

        self.t_sim = 0.0
        # Initialize state with defaults until vehicle connects
        self.state = {"x": [0,0,0], "q": [0,0,0,1], "rotor_speeds": [0,0,0,0]} 

        self.clients: set[WebSocket] = set()
        self.command_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def _create_vehicle_sync(self):
        """Blocking call to instantiate PX4Multirotor"""
        print("Attempting to connect to PX4...", flush=True)
        return PX4Multirotor(quad_params, enable_ground=True, initial_state=None)

    async def reset_sim(self):
        """Async reset that offloads the blocking re-instantiation"""
        print("Resetting Simulation...", flush=True)
        # Use to_thread here so a reset doesn't freeze the websocket/server
        self.vehicle = await asyncio.to_thread(self._create_vehicle_sync)
        self.t_sim = 0.0
        self.state = self.vehicle.initial_state
        print("Simulation Reset Complete", flush=True)

    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.sim_loop())

    async def sim_loop(self):
        while True:
            frame_start = time.perf_counter()

            # 2. Connection Phase
            if self.vehicle is None:
                try:
                    # Offload the blocking init to a worker thread.
                    # This allows the MainThread to process SIGINT while waiting.
                    self.vehicle = await asyncio.to_thread(self._create_vehicle_sync)
                    
                    # Connection successful
                    self.state = self.vehicle.initial_state
                    print("PX4 Connected!", flush=True)
                except asyncio.CancelledError:
                    # Handle shutdown request during connection attempt
                    raise
                except Exception as e:
                    print(f"Connection failed: {e}. Retrying in 1s...", flush=True)
                    await asyncio.sleep(1.0)
                
                # Skip the physics step if we just connected or failed
                continue

            # 3. Physics Phase (Only runs if vehicle is connected)
            try:
                flat = hover_trajectory.update(self.t_sim)
                control_dict = self.controller.update(self.t_sim, self.state, flat)
                self.state = self.vehicle.step(self.state, control_dict, self.dt_target)
                self.t_sim += self.dt_target
            except Exception as e:
                print(f"Simulation Error: {e}")
                # Optional: self.vehicle = None (to trigger a reconnect)

            await self._drain_commands()
            await self.broadcast_state()

            # 4. Timing
            frame_end = time.perf_counter()
            dt_actual = frame_end - frame_start
            sleep_time = self.dt_target - dt_actual
            
            # Ensure sleep_time is non-negative
            await asyncio.sleep(max(0.0, sleep_time))

    async def _drain_commands(self):
        while not self.command_queue.empty():
            cmd = await self.command_queue.get()
            ctype = cmd.get("cmd")
            if ctype == "restart":
                await self.reset_sim() # Await the async reset

    async def broadcast_state(self):
        if not self.clients or self.vehicle is None:
            return

        # Ensure state keys exist before accessing
        try:
            msg = {
                "t_sim": self.t_sim,
                "x": np.asarray(self.state["x"]).tolist(),
                "q": np.asarray(self.state["q"]).tolist(),
                "rotor_speeds": np.asarray(self.state["rotor_speeds"]).tolist()
            }
            data = json.dumps(msg)
        except KeyError:
            return

        disconnected = []
        for ws in self.clients:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.clients.discard(ws)

    # ... register/unregister/handle_ws remain the same ...
    async def register(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)

    async def unregister(self, websocket: WebSocket):
        if websocket in self.clients:
            self.clients.remove(websocket)

    async def handle_ws(self, websocket: WebSocket):
        await self.register(websocket)
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    cmd = json.loads(message)
                    await self.command_queue.put(cmd)
                except json.JSONDecodeError:
                    continue
        except WebSocketDisconnect:
            pass
        finally:
            await self.unregister(websocket)

# --- App Setup ---

sim_server: SimServer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_server
    sim_server = SimServer()
    await sim_server.start()
    try:
        yield
    finally:
        print("Shutting down sim server...", flush=True)
        # 1. Cancel the asyncio task
        if sim_server._task:
            sim_server._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                # Wait briefly for it to clean up, but don't wait forever
                try:
                    await asyncio.wait_for(sim_server._task, timeout=2.0)
                except asyncio.TimeoutError:
                    print("Sim task did not cancel gracefully (thread stuck?)", flush=True)

        print("Forcing process exit...", flush=True)
        # 2. Hard exit to kill any stuck background threads (like the PX4 connection)
        os._exit(0)

app = FastAPI(lifespan=lifespan)

# Add your routes back here
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if sim_server:
        await sim_server.handle_ws(websocket)

config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=False)
server = uvicorn.Server(config)

if __name__ == "__main__":
    print("Starting python sidecar...", flush=True)
    asyncio.run(server.serve())