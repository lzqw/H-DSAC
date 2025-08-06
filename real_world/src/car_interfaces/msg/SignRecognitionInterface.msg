# 交通标识识别话题
# 话题名称：sign_recognition_data
# 话题接口：car_interfaces/msg/SignRecognitionInterface.msg

float64             timestamp       # 时间戳
uint8               id              # 摄像头ID
uint8               signnumber      # 交通标识数量
float32[]           signdata        # 交通标志数据
sensor_msgs/Image   resultimage     # 交通标志识别结果图像
float32             process_time    # 进程处理时间

# 其中signdata格式float32[]为一维数组
# 储存按照[category ,length ,width ,height , x  ,y  ,z  ,v  ,latv ,category  ,length ,width ,height , x ,y ,z ,v ,latv ,……]存储
# 9个长度数据为一组
# int   category    交通标志类别，1限速标识、2停车让行标识
# float length      交通标志长度m，精度为0.01
# float width       交通标志宽度m，精度为0.01
# float height      交通标志高度m，精度为0.01
# float x           交通标志的x坐标[米]，精度为0.01
# float y           交通标志的y坐标[米]，精度为0.01
# float z           交通标志的z坐标[米]，精度为0.01
# float v=0         交通标志径向速度m/s，默认为0，精度为0.01
# float latv=0      交通标志的横向速度m/s，默认为0，精度为0.01
