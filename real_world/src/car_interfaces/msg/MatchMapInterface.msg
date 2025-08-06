# 车辆在地图上的实时位置跟踪话题
# 话题名称：match_map_data
# 话题接口：car_interfaces/msg/MatchMapInterface.msg（自定义）

float64     timestamp               # 时间戳
uint8       special_road            # 0无意义，1进入红绿灯路口范围
float64     stopline_longitude      # 停车线经度
float64     stopline_latitude       # 停车线纬度
bool        cross                   # 是否已过停车线，0否，1是（进入红绿灯区域，若过了停车线，关闭红绿灯识别模块）
bool        light_recognition       # 是否启动红绿灯识别模块，0否，1是
