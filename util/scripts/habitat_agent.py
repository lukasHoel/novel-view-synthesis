#!/usr/bin/env python
# coding: utf-8

import pickle
import random
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import quaternion

import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene.id = settings["scene"]
    
    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },  
    }
    
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)
            
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=1)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=1)
        ),
    }
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def display_sample(observations):
    """Plot RGB, Semantic and Depth images"""
    rgb_obs = observations["color_sensor"]
    semantic_obs = observations["semantic_sensor"]
    depth_obs = observations["depth_sensor"]

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, semantic_img, depth_img]
    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()
    
def get_camera_matrices(position, rotation):
    rotation = quaternion.as_rotation_matrix(rotation)
    
    # Pinv: Agent/Camera pose wrt Habitat WCS
    Pinv = np.eye(4)
    Pinv[0:3, 0:3] = rotation
    Pinv[0:3, 3] = position
    # P: Habitat WCS wrt Agent/Camera
    P = np.linalg.inv(Pinv)

    return P, Pinv

def get_visuals(observations):
    """Returns PIL versions of RGB, Semantic and Depth images, also returns Depth array"""
    rgb_img = observations["color_sensor"]
    rgb_img = Image.fromarray(rgb_img, mode="RGBA")
    
    sem = observations["semantic_sensor"]
    sem_img = Image.new("P", (sem.shape[1], sem.shape[0]))
    sem_img.putpalette(d3_40_colors_rgb.flatten())
    sem_img.putdata((sem.flatten() % 40).astype(np.uint8))
    sem_img = sem_img.convert("RGBA")
    
    dep_arr = observations["depth_sensor"]
    dep_img = Image.fromarray((dep_arr / 10 * 255).astype(np.uint8), mode="L")
    
    return rgb_img, sem_img, dep_img, dep_arr

def collect_all_data(observations, state):
    rgb_img, sem_img, _, dep_arr = get_visuals(observations)
    P, Pinv = get_camera_matrices(state.position, state.rotation)
    return rgb_img, sem_img, dep_arr, Pinv

def split_RT(RT):
    formatter={'float_kind':lambda x: "%.10f" % x}
    R = RT[0:3, 0:3]
    cam_pos = RT[0:3, 3].ravel()
    cam_up = R[:, 1].ravel()  # y=cam_up (already unit)
    cam_dir = R[:, 2].ravel() # z=cam_dir (already unit)
    cam_pos = np.array2string(cam_pos, formatter=formatter, max_line_width=np.inf, separator=", ")
    cam_up = np.array2string(cam_up, formatter=formatter, max_line_width=np.inf, separator=", ")
    cam_dir = np.array2string(cam_dir, formatter=formatter, max_line_width=np.inf, separator=", ")
    return cam_pos, cam_up, cam_dir

def save_data(path, index, rgb, sem, dep, Pinv):
    file_name = "sample_" + str(index)
    rgb = rgb.convert("RGB")
    sem = sem.convert("RGB")
    rgb.save(os.path.join(path, file_name + ".png"))
    sem.save(os.path.join(path, file_name + ".seg.png"))
    np.save(os.path.join(path, file_name + ".depth.npy"), dep)
    
    cam_file_content = "{:<12} = {}';\n"
    cam_pos, cam_up, cam_dir = split_RT(Pinv)
    info = cam_file_content.format("cam_pos", cam_pos)
    info += cam_file_content.format("cam_dir", cam_dir)
    info += cam_file_content.format("cam_up", cam_up)
    with open(os.path.join(path, file_name + ".txt"), 'w+') as f:
        f.write(info)

def query_seg_color(objID):
    # Identify segmentation color for the modified object
    color = d3_40_colors_rgb[OBJID]
    print("RGB (Unnormalized):", color)
    print("RGB (Normalized):", color/255)
    color = np.broadcast_to(color, (256, 256, 3))
    img = Image.fromarray(color, mode="RGB")
    img.show()

class MainWindow(QWidget):
    def __init__(self, sim, agent, state_hist, output_path):
        super().__init__()
        self.sim = sim
        self.agent = agent
        self.state_hist = state_hist
        self.output_path = output_path
        self.action_map = {
            Qt.Key_4: "turn_left",
            Qt.Key_6: "turn_right",
            Qt.Key_8: "look_up",
            Qt.Key_5: "look_down",
            Qt.Key_W: "move_forward",
            Qt.Key_A: "move_left",
            Qt.Key_S: "move_backward",
            Qt.Key_D: "move_right"
        }
        self.initialize()
        
    def get_imageQt(self, observations):
        """Returns Qt versions of RGB, Semantic and Depth images"""
        rgb_img, sem_img, dep_img, _ = get_visuals(observations)
        rgb_img = ImageQt(rgb_img)
        rgb_img = QPixmap.fromImage(rgb_img)
        
        sem_img = ImageQt(sem_img)
        sem_img = QPixmap.fromImage(sem_img)
        
        dep_img = ImageQt(dep_img)
        dep_img = QPixmap.fromImage(dep_img)
        return rgb_img, sem_img, dep_img 
        
    def initialize(self):
        self.title = "Habitat Agent"
        self.top = 0
        self.left = 0
        self.width = 256*3
        self.height = 456
        self.timestep = 0
        
        hbox = QHBoxLayout()
        
        rgb_panel = QFrame()
        rgb_panel.setFrameShape(QFrame.StyledPanel)
        self.rgb_panel = QLabel(rgb_panel)
        
        seg_panel = QFrame()
        seg_panel.setFrameShape(QFrame.StyledPanel)
        self.seg_panel = QLabel(seg_panel)
        
        dep_panel = QFrame()
        dep_panel.setFrameShape(QFrame.StyledPanel)
        self.dep_panel = QLabel(dep_panel)
        
        self.info_panel = info_panel = QPlainTextEdit()
        info_panel.setReadOnly(True)

        split1 = QSplitter(Qt.Horizontal)
        split1.addWidget(rgb_panel)
        split1.addWidget(seg_panel)
        split1.setSizes([256,256])
        
        split2 = QSplitter(Qt.Horizontal)
        split2.addWidget(split1)
        split2.addWidget(dep_panel)
        split2.setSizes([512,256])
        
        split3 = QSplitter(Qt.Vertical)
        split3.addWidget(split2)
        split3.addWidget(info_panel)
        split3.setSizes([256,200])
        hbox.addWidget(split3)
        
        # Render images on respective windows
        observations = self.sim.get_sensor_observations()
        agent_state = self.agent.get_state()
        # P, Pinv = get_camera_matrices(agent_state.position, agent_state.rotation)
        
        rgb, seg, dep = self.get_imageQt(observations)
        self.rgb_panel.setPixmap(rgb)
        self.seg_panel.setPixmap(seg)
        self.dep_panel.setPixmap(dep)

        log = "t: {}, Position: {}, Orientation: {}".format(self.timestep, agent_state.position, agent_state.rotation)
        self.info_panel.appendPlainText(log)
        
        self.setLayout(hbox)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.show()

    def keyPressEvent(self, event):
        key = event.key()
        # Clear logger
        if key == Qt.Key_C:
            self.info_panel.clear()
        
        # Close window
        elif key == Qt.Key_Escape:
            self.close()
        
        # Save current observation
        elif key == Qt.Key_P:
            observations = self.sim.get_sensor_observations()
            agent_state = self.agent.get_state()
            file_index = len(self.state_hist)
            self.state_hist.append(agent_state)
            data = collect_all_data(observations, agent_state)
            save_data(self.output_path, file_index, *data)
            log = "Saving data at t:{}".format(self.timestep)
            self.info_panel.appendPlainText(log)
            
        # TODO: Adjustable speed
        elif key == Qt.Key_Plus:
            agent.agent_config.action_space["move_forward"].actuation.amount += 0.1
        elif key == Qt.Key_Minus:
            agent.agent_config.action_space["move_forward"].actuation.amount -= 0.1
        
        # Take an action
        elif key in self.action_map:
            action = self.action_map[key]
            observations = self.sim.step(action)
            self.timestep += 1
            
            agent_state = self.agent.get_state()
            # P, Pinv = get_camera_matrices(agent_state.position, agent_state.rotation)
            
            rgb, seg, dep = self.get_imageQt(observations)
            self.rgb_panel.setPixmap(rgb)
            self.seg_panel.setPixmap(seg)
            self.dep_panel.setPixmap(dep)
            
            log = "t:{}, Position: {}, Orientation: {}".format(self.timestep, agent_state.position, agent_state.rotation)
            self.info_panel.appendPlainText(log)

def init_sim(sim_settings, start_pos, start_rot):
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    random.seed(sim_settings["seed"])
    sim.seed(sim_settings["seed"])

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(start_pos)               # Agent start position set
    agent_state.rotation = quaternion.quaternion(*start_rot) # Agent start orientation set
    agent.set_state(agent_state)

    return sim, agent, cfg

def main(argv):
    if len(argv) < 4:
        print("Required 4 args: \n(1) scene_ply\n(2) output_path\n(3) start_pos\n(4) start_rot\n")
        exit(-1)

    scene_ply = argv[0]
    output_path = argv[1]
    start_pos = eval(argv[2])
    start_rot = eval(argv[3])

    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    sim_settings = {
        # Spatial resolution of the observations
        "width": 256,
        "height": 256,
        "scene": scene_ply,      # Scene path
        "default_agent": 0,      # Agent ID
        "sensor_height": 0,      # Height of sensors in meters
        "color_sensor": True,    # RGB sensor
        "semantic_sensor": True, # Semantic sensor
        "depth_sensor": True,    # Depth sensor
        "seed": 1,
    }

    sim, agent, cfg = init_sim(sim_settings, start_pos, start_rot)
    state_hist = [] # Keep agent state when a sample is taken

    action_names = list(
        cfg.agents[
            sim_settings["default_agent"]
        ].action_space.keys()
    )

    # Control agent with GUI
    app = QApplication([])
    window = MainWindow(sim, agent, state_hist, output_path)
    window.show()
    app.exec_()

    # Simulation ends:
    # Save trajectory used for sampling
    with open(os.path.join(output_path, "trajectory.txt"), "wb+") as file:
        pickle.dump(state_hist, file)

# Execute only if run as a script
if __name__ == "__main__":
    main(sys.argv[1:])