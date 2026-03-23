import os
import asyncio
import json
import time
import numpy as np
from contextlib import asynccontextmanager
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy.spatial.transform import Rotation as R
import uvicorn

# --- Import your simulation dependencies here ---
from lib.px4_multirotor_edit import PX4Multirotor
from rotorpy.vehicles.px4_sihsim_quadx_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj

hover_trajectory = HoverTraj(x0=np.array([0, 0, 5]))
initial_rotation = R.from_euler('z', 90, degrees=True)
initial_quat = initial_rotation.as_quat()

initial_state = {
    'x': np.zeros(3),
    'v': np.zeros(3),
    'q': initial_quat, # Apply an initial rotation of 90deg so facing north to fit px4
    'w': np.zeros(3),
    'wind': np.zeros(3),
    'rotor_speeds': np.zeros(quad_params['num_rotors'])
}

class SimServer:
    def __init__(self):
        self.vehicle = None 
        self.controller = SE3Control(quad_params)

        self.target_fps = 150.0
        self.dt_target = 1.0 / self.target_fps
        self.t_sim = 0.0
        self.state = {"x": [0,0,0], "q": [0,0,0,1], "rotor_speeds": [0,0,0,0]} 

        # --- Connection State ---
        self.godot_client: WebSocket | None = None
        self.robotics_clients: set[WebSocket] = set()
        
        self.command_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    # --- 1. Restored & Adapted Connection Management ---
    
    async def disconnect(self, websocket: WebSocket):
        """Cleanly removes a websocket from whichever set it belongs to."""
        if websocket == self.godot_client:
            print("Godot Engine disconnected.", flush=True)
            self.godot_client = None
        elif websocket in self.robotics_clients:
            print("A robotics client disconnected.", flush=True)
            self.robotics_clients.remove(websocket)

    async def handle_ws(self, websocket: WebSocket):
        await websocket.accept()
        
        # --- Identification Phase ---
        try:
            # Wait for the first message to identify the client
            first_msg = await websocket.receive_text()
            data = json.loads(first_msg)
            
            if data.get("type") == "handshake" and data.get("source") == "godot":
                print("Godot Engine linked.", flush=True)
                # If an old connection exists, close it first
                if self.godot_client is not None:
                    await self.godot_client.close()
                self.godot_client = websocket
            else:
                print("New robotics client connected.", flush=True)
                self.robotics_clients.add(websocket)
                
                # FIX: Check if the first message has a "type" and isn't just a handshake.
                # If it's an actionable request (like "request_image"), put it in the queue.
                if data.get("type") and data.get("type") != "handshake":
                    await self.command_queue.put(data)
                    
        except Exception as e:
            print(f"Handshake failed: {e}")
            await websocket.close()
            return

        # --- Main Loop with Disconnect Handling ---
        try:
            while True:
                # receive() handles both text and binary automatically
                message = await websocket.receive()

                if "text" in message:
                    try:
                        cmd = json.loads(message["text"])
                        await self.command_queue.put(cmd)
                    except json.JSONDecodeError:
                        print("Received invalid JSON", flush=True)

                elif "bytes" in message:
                    # If Godot sends binary data (image), broadcast to all robotics clients
                    if websocket == self.godot_client:
                        # We use a copy of the set to avoid "Set changed size during iteration" errors
                        # if a client disconnects mid-broadcast
                        for client in list(self.robotics_clients):
                            try:
                                await client.send_bytes(message["bytes"])
                            except Exception:
                                # If sending fails, assume disconnect and remove
                                await self.disconnect(client)

        except WebSocketDisconnect:
            # This catches the normal "client closed connection" event
            pass
        except Exception as e:
            print(f"Socket error: {e}", flush=True)
        finally:
            # This ensures cleanup happens no matter how the loop exits
            await self.disconnect(websocket)

    # --- 2. Simulation Logic (Unchanged) ---

    def _create_vehicle_sync(self):
        print("Attempting to connect to PX4...", flush=True)
        return PX4Multirotor(quad_params, enable_ground=True, initial_state=initial_state)

    async def reset_sim(self):
        print("Resetting Simulation...", flush=True)
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

            if self.vehicle is None:
                try:
                    self.vehicle = await asyncio.to_thread(self._create_vehicle_sync)
                    self.state = self.vehicle.initial_state
                    print("PX4 Connected!", flush=True)
                except Exception:
                    await asyncio.sleep(1.0)
                continue

            try:
                flat = hover_trajectory.update(self.t_sim)
                control_dict = self.controller.update(self.t_sim, self.state, flat)
                self.state = self.vehicle.step(self.state, control_dict, self.dt_target)
                self.t_sim += self.dt_target
            except Exception as e:
                print(f"Sim Error: {e}")

            await self._drain_commands()
            await self.broadcast_state()

            dt_actual = time.perf_counter() - frame_start
            await asyncio.sleep(max(0.0, self.dt_target - dt_actual))

    async def _drain_commands(self):
        while not self.command_queue.empty():
            cmd = await self.command_queue.get()
            
            # Standardized on "type"
            msg_type = cmd.get("type")
            
            # Forward image requests to Godot
            if msg_type == "request_image":
                if self.godot_client:
                    await self.godot_client.send_text(json.dumps(cmd))
            
            elif msg_type == "restart":
                await self.reset_sim()

    async def broadcast_state(self):
        if not self.vehicle: return
        
        try:
            # Prepare state message
            msg = json.dumps({
                "type": "state",
                "data": {
                    "t_sim": self.t_sim,
                    "x": np.asarray(self.state["x"]).tolist(),
                    "v": np.asarray(self.state["v"]).tolist(),
                    "q": np.asarray(self.state["q"]).tolist(),
                    "w": np.asarray(self.state["w"]).tolist(),
                    "rotor_speeds": np.asarray(self.state["rotor_speeds"]).tolist()
                }
            })
            
            # Send to Godot
            if self.godot_client:
                try:
                    await self.godot_client.send_text(msg)
                except Exception:
                    await self.disconnect(self.godot_client)
            
            # Send to Robotics Clients
            for ws in list(self.robotics_clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    await self.disconnect(ws)
                    
        except Exception:
            pass

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
        if sim_server._task: sim_server._task.cancel()
        os._exit(0)

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if sim_server:
        await sim_server.handle_ws(websocket)

if __name__ == "__main__":
    asyncio.run(uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000)).serve())