# 导航数据话题
# 话题名称：gps_data  
# 话题接口：car_interfaces/msg/GpsInterface.msg

float64     timestamp       # 时间戳
uint8       id              # 导航ID
float32     yaw             # 偏航角（0-359.99）
float32     pitch           # 俯仰角（-90-90）
float32     roll            # 横滚角（-180-180）
float32     wx              # 角速度x
float32     wy              # 角速度y
float32     wz              # 角速度z
float32     ax              # 加速度x
float32     ay              # 加速度y
float32     az              # 加速度z
float64     longitude       # 经度
float64     latitude        # 纬度
float64     height          # 高度
float32     eastvelocity    # 东向速度
float32     northvelocity   # 北向速度
float32     skyvelocity     # 天向速度
float32     process_time    # 进程处理时间

# 规定地球坐标系，东经为正，西经为负，北纬为正，南纬为负
# 规定航向角0°为东经方向，逆时针旋转为正，范围为0-360°
# 规定车辆坐标系车辆前方为Y，右侧为X，上方为Z
# 规定车辆坐标系原点为车辆物理中心点

