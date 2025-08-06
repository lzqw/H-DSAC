# 决策状态机数据话题
# 话题名称: car_decision_data
# 话题接口: car_interfaces/msg/CarDecisionInterface.msg

float64     timestamp       # 数据时间戳
bool        stop            # 是否停车，0继续前进，1停车（判断依据：障碍物位置）
bool        speed           # 是否限速，0否，1是
float32     maxspeed        # 路段限速最大速度，单位为m/s（依据：限速标识）
float32     process_time    # 进程处理时间
