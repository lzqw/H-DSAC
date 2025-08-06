# 车辆底盘数据
# 话题名称：car_ori_data
# 话题接口：car_interfaces/msg/CarOriInterface.msg（自定义）

float64     timestamp               # 时间戳
uint8       id                      # 车辆ID
float32     carspeed                # 车辆速度，单位为m/s
float32     steerangle              # 车辆转角，左转为正，右转为负
int8        gearpos                 # 车辆档位信号（01：驻车，02：空挡（N），03：前进（D），04：后退（R），05：无效）
float32     braketq                 # 制动量（-50-50nm）
uint8       parkingstate            # 制动状态（00：驻车状态，01：驻车释放状态）
uint8       soc                     # 电池当前SOC（0-100）
uint8       batteryvol              # 电池电压（0-55V）
uint16      batterydischargecur     # 电池放电电流（0-100A）
uint8       car_run_mode            # 车辆运行模式： 0： ACU控制  1：自动驾驶  2： 急停模式
uint8       throttle_percentage     # 油门踏板开度： 取值0～100
uint8       braking_percentage      # 制动踏板开度： 取值0～100
bool        left_light              # 左转向灯状态：0：关闭，1：打开
bool        right_light             # 右转向灯状态：0：关闭，1：打开
bool        reversing_light         # 倒车灯状态：0：关闭，1：打开
bool        speaker                 # 喇叭状态：0：关闭，1：打开
bool        start_button            # 启动按钮状态：0：按键无效，1：按键有效
bool        stop_button             # 急停按钮状态：0：按键无效，1：按键有效
uint8       state                   # 设备状态，0：状态正常，1：电池箱报警；2：电机控制器报警
uint8       error                   # 错误码；电池箱报警：1：单体过压或欠压，2：放电电流异常，3：电压报警，4：电池温度报警，5：电池SOC过低。电机控制器报警：1：转向电机控制器故障，2：驱动电机控制器故障
float32     process_time            # 进程处理时间
