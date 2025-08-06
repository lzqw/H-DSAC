import datetime
import math
import sys
import os
import time
import copy
import json
from typing import List
import numpy as np
import rclpy
from rclpy.node import Node
from car_interfaces.msg import *
import can
import torch
ros_node_name = "car_rl"
sys.path.append(os.getcwd() + "/src/utils/")
sys.path.append(os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name))
sys.path.append(os.getcwd() + "/src/%s/%s/utils"%(ros_node_name, ros_node_name))
print(os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name))
import tjitools
import yaml
from initialization import create_alg,create_sampler,create_buffer
from common_utils import change_type
from collections import deque
from dpvp_rl import DPVP
from torch.utils.tensorboard import SummaryWriter
from tensorboard_setup import tb_tags,add_scalars
from dpvp import DPVPPolicy
from dpvp_buffer import DPVPBuffer

OBS_DIM=445-1
PI = 3.1415926535
    
class CarRL(Node):
    def __init__(self):
        super().__init__(ros_node_name)

        # define subscribers
        self.subSurrounding = self.create_subscription(SurroundingInfoInterface, "surrounding_info_data", self.sub_callback_surrounding, 1)
                 
        # define publishers
        self.pubCarRL = self.create_publisher(CarRLInterface, "car_rl_data", 10)
        self.timerCarRL= self.create_timer(0.1, self.pub_callback_car_rl)
        self.carControlBus = can.interface.Bus(channel='can1', bustype='socketcan')
        self.takeover_recorder = deque(maxlen=2000)

        #init dpvp
        self.init_dpvp()
        self.iteration=0

        self.rcvMsgSurroundingInfo=None
        
        tjitools.ros_log(self.get_name(), 'Start Node:%s'%self.get_name())

    def init_dpvp(self):
        with open(os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name)+"/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            self.config = config
        self.config['obsv_dim']=OBS_DIM
        self.config['act_dim']=2
        self.config["action_high_limit"]=np.array([1.0, 1.0])
        self.config["action_low_limit"]=np.array([-1.0, -1.0])
        self.config["action_type"]="continu"
        # if self.config["save_folder"] is None:
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.dirname(dir_path)
        dir_path="/home/nvidia/AutoDrive"
        self.config["save_folder"] = os.path.join(
            dir_path + "/results/",
            "DPVP" +str("_")+ "CAR",
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.config["save_folder"], exist_ok=True)
        os.makedirs(self.config["save_folder"] + "/apprfunc", exist_ok=True)
        with open(self.config["save_folder"] + "/config.json", "w", encoding="utf-8") as f:
            json.dump(change_type(copy.deepcopy(self.config)), f, ensure_ascii=False, indent=4)

        self.dpvp = DPVP(**self.config)
        self.networks=self.dpvp.networks
        print("DPVP Policy is created.")
        self.buffer = DPVPBuffer(**self.config)
        print("DPVP Buffer is created.")

        self.replay_batch_size = self.config["replay_batch_size"]
        self.max_iteration = self.config["max_iteration"]
        self.update_interval = self.config.get("update_interval", 1)
        self.log_save_interval = self.config["log_save_interval"]
        self.apprfunc_save_interval = self.config["apprfunc_save_interval"]
        self.save_folder = self.config["save_folder"]
        self.policy_update_time=int(self.config["policy_update_time"])

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.alg_tb_dict={}

        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        self.use_gpu = self.config["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        

    def pub_callback_car_rl(self):
        """
        Callback of publisher (timer), publish the topic car_rl_data.
        :param None.
        """
        msgCarRL= CarRLInterface()
        now_ts = time.time()
        msgCarRL.timestamp = now_ts

        if self.rcvMsgSurroundingInfo is not None:
            self.iteration+=1
            print("iter:",self.iteration)
            if self.rcvMsgSurroundingInfo.gearpos!=2:
                batch_data=self.sample()
                self.buffer.add_batch(batch_data)
            print(self.buffer.human_size,",",self.config["buffer_warm_size"])
            if self.iteration % self.update_interval== 0 and self.buffer.human_size > self.config["buffer_warm_size"]:
                print("train begin!!!!")
                self.train_policy()
                print("policy_trained")
            if self.iteration % self.log_save_interval == 0:
                print("Iter = ", self.iteration)
                add_scalars(self.alg_tb_dict, self.writer, step=self.iteration)

                add_scalars({"takeover_rate":np.mean(np.array(self.takeover_recorder)) * 100},self.writer, step=self.iteration)

            if self.iteration % self.apprfunc_save_interval == 0:
                self.save_apprfunc()
                
            msgCarRL.process_time = time.time() - now_ts
            self.pubCarRL.publish(msgCarRL)
        tjitools.ros_log(self.get_name(), 'Publish car_rl msg !!!')

    def save_apprfunc(self):
        save_data = {
            "state_dict": self.networks.state_dict(),  # Save all model parameters
            "mean_std1_behavior": self.networks.mean_std1_behavior,
            "mean_std2_behavior": self.networks.mean_std2_behavior,
            "mean_std1_novice": self.networks.mean_std1_novice,
            "mean_std2_novice": self.networks.mean_std2_novice,
        }
        torch.save(
            save_data,
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    def train_policy(self):

        if self.buffer.human_size > self.replay_batch_size and self.buffer.size > self.replay_batch_size:
            replay_samples_agent= self.buffer.sample_batch(int(self.replay_batch_size/2))
            replay_samples_human = self.buffer.sample_human_batch(int(self.replay_batch_size/2))
            replay_samples = {k: torch.cat((replay_samples_agent[k], replay_samples_human[k]), dim=0) for k in
                                replay_samples_agent.keys()}
        elif self.buffer.human_size > self.replay_batch_size:
            replay_samples = self.buffer.sample_human_batch(self.replay_batch_size)
        elif self.buffer.size > self.replay_batch_size:
            replay_samples = self.buffer.sample_batch(self.replay_batch_size)
        else:
            return
        
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()
        for _ in range(self.policy_update_time):
            self.alg_tb_dict = self.dpvp.local_update(replay_samples, self.iteration)


    def sample(self):
        """
        Sample the data from the environment.
        :param None.
        :return: The sampled data.
        """
        state=self.get_state()
        batch_data = []

        self.episode = {}
        batch_obs = torch.from_numpy(
            np.expand_dims(state, axis=0).astype("float32")
        ).to('cuda')

        with torch.no_grad():
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)

            action, logp = action_distribution.sample()

            action = action.to('cpu').detach()[0].numpy()
            logp = logp.to('cpu').detach()[0].numpy()

            action = np.array(action)
            action_clip = action.clip(
                self.config["action_low_limit"], self.config["action_high_limit"]
            )


        self.send_action(action_clip)
        state_next=self.get_state()

        intervention=1-self.rcvMsgSurroundingInfo.car_run_mode
        self.takeover_recorder.append(intervention)


        action_behavior=self.reverse_process_action(self.rcvMsgSurroundingInfo.throttle_percentage,
                                                    self.rcvMsgSurroundingInfo.braking_percentage,
                                                    self.rcvMsgSurroundingInfo.steerangle)

        #TODO: whether the episode is done
        done=False

        action_novice=action
        _stop_td = intervention
        data = [
            state.copy(),
            action_behavior,
            state_next.copy(),
            done,
            logp,

            action_novice,
            intervention,
            np.float32(1-_stop_td)
        ]
        batch_data.append(tuple(data))

        return batch_data






    def is_arrived(self, current_pos: List[float], target_pos: List[float], dist: float):
        """
        Check if the car has arrived at the target position.
        :param current_pos: The current position of the car.
        :param target_pos: The target position of the car.
        :param dist: The distance threshold to determine if the car has arrived.
        :return: True if the car has arrived, False otherwise.
        """
        #TODO: how to dientity the car has arrived
        return np.linalg.norm(np.array(current_pos) - np.array(target_pos)) < dist
    
    def get_state(self):
        """
        Get the state of the car.
        :param None.
        :return: The state of the car.
        """
        # TODO : get the state of the car, need to range to 0-1
        state=[]

        errorYaw=self.rcvMsgSurroundingInfo.error_yaw/PI
        errorYaw=np.clip(errorYaw,-1.0,1.0)
        state.append(errorYaw)

        errorDistance=self.rcvMsgSurroundingInfo.error_distance/5.0
        errorDistance=np.clip(errorDistance,-1.0,1.0)
        state.append(errorDistance)

        carspeed=self.rcvMsgSurroundingInfo.carspeed/15.0
        carspeed=np.clip(carspeed,0.0,1.0)
        state.append(carspeed)



        turn_state=self.rcvMsgSurroundingInfo.turn_signals
        # state.append(steUI
        turn_state=(turn_state+1)/2
        state.append(turn_state)


        state.extend(list(self.rcvMsgSurroundingInfo.surroundinginfo))

        path=list(self.rcvMsgSurroundingInfo.path_rfu)
        path=np.array(path)/30.0
        path=np.clip(path,-1.0,1.0)
        state.extend(list(path))
        return state
    

    def reverse_process_action(self,throttle_percentage,braking_percentage,steering_angle):
        """
        Convert an action vector [x, y] in range [-1, 1] to:
        - throttle_percentage: uint8 (0~100)
        - braking_percentage: uint8 (0~100)
        - steering_angle: float32 (-380~380)
        
        :param action: A list or tuple [x, y], where
                    x in [-1, 1]: throttle/brake
                        (x>0 => throttle, x<0 => brake)
                    y in [-1, 1]: steering
                        (maps to -380~380 in float32)
        :return: (throttle_percentage, braking_percentage, steering_angle)
                throttle_percentage: int (0~100)
                braking_percentage:  int (0~100)
                steering_angle:      float32 (-380~380)
        """
        # print("get_brake:",braking_percentage)
        print("get_throttle:",throttle_percentage)  
        if braking_percentage != 0:
            x= -float(braking_percentage/100)
        else:
            x= float(throttle_percentage/100)
        x=np.clip(x,-1.0,1.0)
        
        y= steering_angle/200.0
        y=np.clip(y,-1.0,1.0)

        return [x,y]

    def process_action(self,action):
        """
        Convert an action vector [x, y] in range [-1, 1] to:
        - throttle_percentage: uint8 (0~100)
        - braking_percentage: uint8 (0~100)
        - steering_angle: float32 (-380~380)
        
        :param action: A list or tuple [x, y], where
                    x in [-1, 1]: throttle/brake
                        (x>0 => throttle, x<0 => brake)
                    y in [-1, 1]: steering
                        (maps to -380~380 in float32)
        :return: (throttle_percentage, braking_percentage, steering_angle)
                throttle_percentage: int (0~100)
                braking_percentage:  int (0~100)
                steering_angle:      float32 (-380~380)
        """
        x = action[0]  # throttle/brake control
        y = action[1]  # steering control

        # Determine throttle or brake
        if x > 0:
            throttle_percentage = int(round(x * 100))  # 0~100
            braking_percentage = 0
        else:
            throttle_percentage = 0
            braking_percentage = int(round(abs(x) * 100))  # 0~100

        steering_angle = y * 200

        return throttle_percentage, braking_percentage, steering_angle


    def send_action(self, action):
        """
        Send the action to the car.
        :param action: The action to be sent to the car.
        :return: None.
        """
        throttle_percentage, braking_percentage, steering_angle = self.process_action(action)
        print("input_brake:",braking_percentage)
        gearpos=0x03
        enableSignal = 1
        ultrasonicSwitch = 0
        dippedHeadlight = 0
        contourLamp = 0

        if braking_percentage!=0:
            brakeEnable = 1
        else:
            brakeEnable = 0
        alarmLamp = 0
        turnSignalControl = 0
        appInsert = 0

        Byte7 = appInsert<<7 | turnSignalControl<<1 | alarmLamp
        Byte6 = braking_percentage <<1 | brakeEnable

        Byte5=(int(steering_angle)&0xFF00)>>8
        Byte4=int(steering_angle) & 0x00FF

        Byte3=0
        Byte2=((throttle_percentage*10)&0xFF00)>>8
        Byte1=(throttle_percentage*10) & 0x00FF
        Byte0=gearpos<<6 | enableSignal<<5 | ultrasonicSwitch<<4 | dippedHeadlight<<1 | contourLamp

        canData=[Byte0,Byte1,Byte2,Byte3,Byte4,Byte5,Byte6,Byte7]
        carControlMsg = can.Message(arbitration_id = 0x210, data=canData, extended_id = False)
        self.carControlBus.send(carControlMsg)

    def test_control(self,action=[0,1]):
        """
        Test the control of the car.
        :param action: The action to be sent to the car.
        :return: None.
        """
        self.send_action(action)
        tjitools.ros_log(self.get_name(), 'Test control !!!')

    def sub_callback_surrounding(self, msgSurroundingInfo:SurroundingInfoInterface):
        """
        Callback of subscriber, subscribe the topic fusion_data.
        :param msgFusion: The message heard by the subscriber.
        """
        self.rcvMsgSurroundingInfo = msgSurroundingInfo
        pass

def main():
    rclpy.init()
    rosNode = CarRL()
    rclpy.spin(rosNode)
    rclpy.shutdown()

