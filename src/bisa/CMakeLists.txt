cmake_minimum_required(VERSION 3.5)
project(bisa)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(osqp_vendor QUIET)
find_package(osqp REQUIRED)

# 메시지 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LapInfo.msg"
  "msg/MPCPerformance.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# ================================================================
# mpc_controller_cpp (라이브러리)
# ================================================================
add_library(mpc_controller_cpp SHARED
  src/mpc_controller_cpp.cpp
)
ament_target_dependencies(mpc_controller_cpp
  rclcpp geometry_msgs nav_msgs
)
target_include_directories(mpc_controller_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  /usr/include/eigen3
)
target_link_libraries(mpc_controller_cpp
  Eigen3::Eigen
  osqp::osqp
)

# ================================================================
# ltv_mpc_core (라이브러리)
# ================================================================
add_library(ltv_mpc_core SHARED
  src/ltv_mpc/ltv_model.cpp
  src/ltv_mpc/ltv_cost.cpp
  src/ltv_mpc/ltv_solver.cpp
  src/ltv_mpc/ltv_mpc.cpp
)
ament_target_dependencies(ltv_mpc_core
  rclcpp geometry_msgs nav_msgs
)
target_include_directories(ltv_mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  /usr/include/eigen3
)
target_link_libraries(ltv_mpc_core
  Eigen3::Eigen
  osqp::osqp
)

# ================================================================
# mpc_path_tracker_cpp (실행 파일)
# ================================================================
add_executable(mpc_path_tracker_cpp
  src/mpc_path_tracker_cpp.cpp
)
ament_target_dependencies(mpc_path_tracker_cpp
  rclcpp std_msgs geometry_msgs nav_msgs
)
target_link_libraries(mpc_path_tracker_cpp
  ltv_mpc_core
  mpc_controller_cpp
)
target_include_directories(mpc_path_tracker_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  /usr/include/eigen3
)

# ================================================================
# local_path_pub_cpp (실행 파일)
# ================================================================
add_executable(local_path_pub_cpp
  src/local_path_pub_cpp.cpp
)
ament_target_dependencies(local_path_pub_cpp
  rclcpp std_msgs geometry_msgs nav_msgs visualization_msgs tf2 tf2_geometry_msgs
)
target_include_directories(local_path_pub_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 메시지 의존성 설정
if(TARGET ${PROJECT_NAME})
  rosidl_target_interfaces(local_path_pub_cpp
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  rosidl_target_interfaces(mpc_path_tracker_cpp
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
endif()

# ================================================================
# 설치 (Install)
# ================================================================
install(PROGRAMS
  scripts/hdmap_visualizer.py
  scripts/global_path_pub.py
  scripts/global_path_pub_multi.py
  scripts/dijkstra_planner.py
  scripts/mgeo_class_defs.py
  scripts/hdmap_generator.py
  scripts/analyze_mpc_performance.py
  scripts/cav_path_error_gui.py
  scripts/cav_runtime_log_gui.py
  scripts/speed_data_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  local_path_pub_cpp
  mpc_path_tracker_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  mpc_controller_cpp
  ltv_mpc_core
  EXPORT mpc_controller_cpp
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY hdmap_data
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
