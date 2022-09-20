cmake_minimum_required(VERSION 2.8.3)
project(ridgeback_description)

find_package(catkin REQUIRED COMPONENTS roslaunch)

catkin_package()

install(DIRECTORY meshes launch urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS scripts/env_run
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(launch/description.launch)
endif()
