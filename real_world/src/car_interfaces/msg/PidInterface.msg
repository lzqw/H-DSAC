# 车辆控制数据话题
# 话题名称: pid_data
# 话题接口: car_interfaces/msg/PidInterface.msg

float64     timestamp               # 数据帧时间戳
float32     velocity                # 车辆速度m/s
float32     angle                   # 车辆转角，方向盘转角 
uint8       gear                    # 01：驻车；02：空挡（N）；03：前进（D）；04：后退（R）；05：无效；
uint8       throttle_percentage     # 油门踏板开度（0-100%）   取值：0-100
uint8       braking_percentage      # 刹车踏板开度（0-100%） 取值： 0-100
float32     turn_signals            #转向灯 -1：左转，0：直行，1：右转
float32     process_time            # 进程处理时间

