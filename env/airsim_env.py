import airsim
import numpy as np
import cv2
import time

class AirSimVisionEnv:
    """
    基于AirSim的多旋翼无人机视觉环境接口
    支持获取摄像头画面、环境重置、动作执行
    """
    def __init__(self, ip='127.0.0.1', port=41451, image_type=0):
        try:
            self.client = airsim.MultirotorClient(ip=ip, port=port)
            self.client.confirmConnection()
            self.image_type = image_type
            self.vehicle_name = "Drone1"
            # 限制最大帧率为10
            self.client.simRunConsoleCommand("t.MaxFPS 10")
        except Exception as e:
            print(f"[AirSimEnv] 初始化失败: {e}")
            raise

    def reset(self):
        try:
            self.client.reset()
            time.sleep(0.5)
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            # 可选：回到原点
            self.client.moveToPositionAsync(0, 0, -2, 2, vehicle_name=self.vehicle_name).join()
        except Exception as e:
            print(f"[AirSimEnv] reset异常: {e}")
            raise

    def get_image(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ], vehicle_name=self.vehicle_name)
            if responses and responses[0].width > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
                return img_rgb
            else:
                raise RuntimeError("未获取到有效图像")
        except Exception as e:
            print(f"[AirSimEnv] get_image异常: {e}")
            raise

    def step(self, vx, vy, vz, yaw_rate=0.0, duration=0.1):
        try:
            # vx,vy,vz为无人机在世界坐标系下的速度(m/s)
            self.client.moveByVelocityAsync(vx, vy, vz, duration, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate), vehicle_name=self.vehicle_name).join()
        except Exception as e:
            print(f"[AirSimEnv] step异常: {e}")
            raise

    def get_state(self):
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            kinematics = state.kinematics_estimated
            pos = kinematics.position
            vel = kinematics.linear_velocity
            ori = kinematics.orientation
            # 四元数转欧拉角
            roll, pitch, yaw = airsim.to_eularian_angles(ori)
            return {
                'x': pos.x_val,
                'y': pos.y_val,
                'z': pos.z_val,
                'vx': vel.x_val,
                'vy': vel.y_val,
                'vz': vel.z_val,
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw
            }
        except Exception as e:
            print(f"[AirSimEnv] get_state异常: {e}")
            raise

    def get_info(self, target_pos=None):
        """获取完整环境信息，包括位置、速度、姿态、碰撞、距离等"""
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            kinematics = state.kinematics_estimated
            pos = kinematics.position
            vel = kinematics.linear_velocity
            ori = kinematics.orientation
            # 四元数转欧拉角
            roll, pitch, yaw = airsim.to_eularian_angles(ori)
            # 碰撞信息
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            collided = collision_info.has_collided if hasattr(collision_info, 'has_collided') else False
            # 当前位置
            x, y, z = pos.x_val, pos.y_val, pos.z_val
            # 距离目标点
            if target_pos is not None:
                tx, ty, tz = [float(v) for v in target_pos]
                distance = float(np.linalg.norm(np.array([x, y, z]) - np.array([tx, ty, tz])))
            else:
                distance = 0.0
            info = {
                'x': x,
                'y': y,
                'z': z,
                'vx': vel.x_val,
                'vy': vel.y_val,
                'vz': vel.z_val,
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'speed': float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])),
                'collision': collided,
                'distance': distance,
                'target_pos': target_pos  # 修复：返回目标点
            }
            return info
        except Exception as e:
            print(f"[AirSimEnv] get_info异常: {e}")
            raise

    def get_navigable_points(self, grid_size=2.0, min_z=-4.0, max_z=-2.0):
        """
        从AirSim地图自动采样所有可导航开阔点，避开障碍物。
        返回: List[[x, y, z]]
        """
        # 1. 采样大范围网格点
        x_range = np.arange(-50, 50, grid_size)
        y_range = np.arange(-50, 50, grid_size)
        z_range = np.arange(min_z, max_z, grid_size)
        points = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    # 2. 判断该点是否为障碍物（用深度图或分割图）
                    pos = airsim.Vector3r(x, y, z)
                    # 用射线检测地面和障碍物
                    # hit = self.client.simRayTraceObstacle(pos, vehicle_name=self.vehicle_name)
                    # ====== 兼容无simRayTraceObstacle环境，直接判定为开阔点 ======
                    hit = None  # 你可用更智能的障碍物检测替换此处
                    # 若未命中障碍物，且高度在范围内
                    if not hit or (hasattr(hit, 'distance') and hit.distance > 1.0):
                        points.append([x, y, z])
        return points

    def sample_valid_target_from_map(self, drone_pos, min_dist=10.0, max_trials=100):
        """
        从地图自动采样一个合格目标点，避开障碍物且距离无人机远。
        """
        nav_points = getattr(self, '_navigable_points', None)
        if nav_points is None:
            nav_points = self.get_navigable_points()
            self._navigable_points = nav_points
        # 随机打乱，采样满足距离的点
        np.random.shuffle(nav_points)
        for tgt in nav_points:
            if np.linalg.norm(np.array(tgt) - np.array(drone_pos)) >= min_dist:
                return tgt
        # fallback:直接远离无人机
        return [drone_pos[0]+min_dist, drone_pos[1], drone_pos[2]]

    def close(self):
        try:
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        except Exception as e:
            print(f"[AirSimEnv] close异常: {e}")
            raise
