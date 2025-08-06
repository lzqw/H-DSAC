# 路径规划起终点话题
# 话题名称：net_traffic_sign_data
# 话题接口：car_interfaces/msg/NetTrafficSignInterface.msg

float64     timestamp       # 时间戳
int8        lightnumber     # 红绿灯数量
int8[]      lightstate      # 红绿灯状态，先ID后状态，数量进行追加, 关/红/绿/黄 – 00/01/02/03, 举例[10,0,8,1,9,2,7,3]，10号关闭，8号红灯，9号绿灯，7号黄灯
int8        etcnumber       # 闸机数量
int8[]      etcstate        # 闸机状态，先ID后状态，数量进行追加, 0关1开, 举例[10,0,8,1,9,1,7,0]，10号关闭，8号开，9号开，7号关闭
int8        parkingnumber   # 停车位数量
int8[]      parkingstate    # 停车位状态，先ID后状态，数量进行追加, 无车/有车 - 00/01, 举例[10,0,8,1,9,1]，10号车位无车关闭，8号车位有车，9号车位有车
float32     process_time    # 进程处理时间
