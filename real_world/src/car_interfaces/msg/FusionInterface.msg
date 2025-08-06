# 融合数据话题
# 话题名称：fusion_data  
# 话题接口：car_interfaces/msg/FusionInterface.msg

float64     timestamp               # 时间戳
uint8       id                      # 车辆ID
float32     carlength               # 车长m
float32     carwidth                # 车宽m
float32     carheight               # 车高m
float32     centeroffset            # 车道中心线偏移距离，默认0
uint8       signnumber              # 交通标识数量
float32[]   signdata                # 交通标志数据
float32[]   lightdata               # 红绿灯数据
float32[]   obstacledata            # 障碍物数据
float32     yaw                     # 偏航角（0-359.99）
float32     pitch                   # 俯仰角（-90-90）
float32     roll                    # 横滚角（-180-180）
float32     wx                      # 角速度x
float32     wy                      # 角速度y
float32     wz                      # 角速度z
float32     ax                      # 加速度x
float32     ay                      # 加速度y
float32     az                      # 加速度z
float64     longitude               # 经度
float64     latitude                # 纬度
float64     height                  # 高度
float32     eastvelocity            # 东向速度
float32     northvelocity           # 北向速度
float32     skyvelocity             # 天向速度
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

# 其中signdata格式float32[]为一维数组
# 储存按照[category ,length ,width ,height , x  ,y  ,z  ,v  ,latv ,category  ,length ,width ,height , x ,y ,z ,v ,latv ,……]存储
# 9个长度数据为一组
# int   category    交通标志类别，需自行定义类别，务必做好记录
# float length      交通标志长度m，精度为0.01
# float width       交通标志宽度m，精度为0.01
# float height      交通标志高度m，精度为0.01
# float x           交通标志的x坐标[米]，精度为0.01
# float y           交通标志的y坐标[米]，精度为0.01
# float z           交通标志的z坐标[米]，精度为0.01
# float v=0         交通标志径向速度m/s，默认为0，精度为0.01
# float latv=0      交通标志的横向速度m/s，默认为0，精度为0.01

# 其中lightdata格式float32[]为一维数组
# 储存按照[x,y,z,leftcolor,straightcolor,rightcolor,……]存储
# 6个长度数据为一组
# float x               红绿灯x坐标[米]，精度为0.01
# float y               红绿灯y坐标[米]，精度为0.01
# float z               红绿灯z坐标[米]，精度为0.01
# int   leftcolor       左转灯状态，01：红，02：黄，03：绿，04：关闭，关闭时代表没有左转灯
# int   straightcolor   直行灯状态，01：红，02：黄，03：绿，04：关闭，
# int   rightcolor      右转灯状态，01：红，02：黄，03：绿，04：关闭，关闭时代表没有右转灯
# 一般情况下，如果左转/右转/左右转和直行灯同时存在，无论什么颜色状态，这两个灯都是常亮的。如果信号灯只有直行灯，那用左/右转灯关闭代表当前路口只有直行灯，直行灯绿色时既可以直行也可以左转。

# 其中obstacledata格式float32[]为一维数组
# 储存按照[category ,length ,width ,height , x  ,y  ,z  ,v  ,latv ,category  ,length ,width ,height , x ,y ,z ,v ,latv ,……]存储
# 9个长度数据为一组
# int   category    障碍物类别， 0点目标；1小汽车；2卡车/客车；3行人；4摩托车/电动车；5自行车；6宽大目标（如墙面）；7reserved保留； 8traffic light； 9stop sign
# float length      障碍物长度m，精度为0.01
# float width       障碍物宽度m，精度为0.01
# float height      障碍物高度m，精度为0.01
# float x           障碍物的x坐标[米]，精度为0.01
# float y           障碍物的y坐标[米]，精度为0.01
# float z           障碍物的z坐标[米]，精度为0.01
# float v=0         障碍物径向速度m/s，精度为0.01
# float latv=0      障碍物的横向速度m/s，精度为0.01
