cmake_minimum_required(VERSION 3.5)
project(car_interfaces)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(msg_files
  # Camera
  "msg/CameraOriInterface.msg"
  "msg/LaneRecognitionInterface.msg"
  "msg/SignRecognitionInterface.msg"
  "msg/LightRecognitionInterface.msg"
  "msg/CameraObstacleInterface.msg"
  "msg/CameraStateInterface.msg"
  # LiDAR
  "msg/LidarOriInterface.msg"
  "msg/LidarObstacleInterface.msg"
  "msg/LidarStateInterface.msg"
  # Radar
  "msg/RadarObstacleInterface.msg"
  "msg/RadarStateInterface.msg"
  # Sonic
  "msg/SonicObstacleInterface.msg"
  "msg/SonicStateInterface.msg"
  # Localization
  "msg/GpsInterface.msg"
  "msg/ImuInterface.msg"
  "msg/GpsStateInterface.msg"
  "msg/ImuStateInterface.msg"
  "msg/MatchMapInterface.msg"
  # VCU
  "msg/CarOriInterface.msg"
  "msg/CanStateInterface.msg"
  # Fusion
  "msg/FusionInterface.msg"
  # Decision
  "msg/CarDecisionInterface.msg"
  # Planning
  "msg/HmiStartEndPointInterface.msg"
  "msg/GlobalPathPlanningInterface.msg"
  "msg/LocalPathPlanningInterface.msg"
  # Control
  "msg/PidInterface.msg"
  # Master
  "msg/RegulatorInterface.msg"
  # Network
  "msg/NetStartEndPointInterface.msg"
  "msg/NetTrafficSignInterface.msg"
  # HMI
  "msg/HmiOtherFunctionsInterface.msg"

  "msg/SurroundingInfoInterface.msg"
  "msg/CarRLInterface.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES sensor_msgs
)
ament_export_dependencies(rosidl_default_runtime)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
