# Dynamixel SDK Release Notes

3.7.31 (2020-07-13)
-------------------
* ROS 1 Noetic Ninjemys support
* 3x faster getError member function of GroupSyncRead Class
* Contributors: developer0hye, Zerom, Will Son

3.7.21 (2019-09-06)
-------------------
* Fixed buffer overflow bug (rxpacket size)
* Fixed typo in the package.xml and header files

3.7.11 (2019-08-19)
-------------------
* Updated C lib and DLL file
* Changed C# / win32 / protocol_combined output path
* Fixed "protocol_combined" example bug
* Fixed typo in bulk_read_write.py

3.7.0 (2019-01-03)
------------------
* Added clear instruction [#269](https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/269)
* Removed busy waiting for rxPacket()
* Fixed addStuffing() function (reduced stack memory usage)
* Fixed memory issues [#268](https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/268)
* Fixed the broadcast ping bug in dxl_monitor

3.6.2 (2018-07-17)
------------------
* Added python modules for ROS to ros folder
* Moved cpp library files for ROS to ros folder
* Created an ROS package separately `#187 <https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/187>`_
* Modified the e-Manual address to emanual.robotis.com

3.6.1 (2018-06-14)
------------------
* Removed printTxRxResult(), printRxPacketError() `#193 <https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/193>`_
* Removed cache files

3.6.0 (2018-03-16)
------------------
* Replaced: DynamixelSDK Python as a native language (Python 2 and 3 for Windows, Linux, Mac OS X) #93 #122 #147 #181 #182 #185
* Added: CONTRIBUTING.md added
* Changes: ISSUE_TEMPLATE.md modified
* Changes: C++ version - SyncRead / BulkRead - getError functions added
* Changes: Deprecated functions removed
* Fixes: DynamixelSDK MATLAB 2017 - new typedef (int8_t / int16_t / int32_t) applied in robotis_def.h #161 #179
* Fixes: Added missing header file for reset and factory_reset examples #167

3.5.4 (2017-12-01)
------------------
* Added: Deprecated is now being shown by attributes #67 #107
* Fixes: DynamixelSDK ROS Indigo Issue - target_sources func in CMake
* Fixes: Bug in protocol1_packet_handler.cpp, line 222 checking the returned Error Mask #120
* Fixes: Packet Handlers - array param uint8_t to uint16_t to avoid closure loop when the packet is too long to be in uint8_t appropriately
* Fixes: Group Syncwrite using multiple ports in c library issue solved (test code is also in this issue bulletin) #124
* Fixes: Support getting of time on MacOSX/XCode versions that doesn't support (CLOCK_REALTIME issue) #141 #144
* Changes: DynamixelSDK Ubuntu Linux usb ftdi latency timer fix issue - changes the default latency timer as 16 ms in all OS, but some about how to change the latency timer was commented in the codes (now the latency timer should be adjusted by yourself... see port_handler_linux source code to see details) #116

3.5.3 (2017-10-30)
------------------
* Fixes: DynamixelSDK ROS Kinetic Issue - ARM - Debian Jessie solved by replacing target_sources func in CMake to set_property #136

3.5.2 (2017-09-18)
------------------
* Recover: Check if the id of rxpacket is the same as the id of txpacket (c++) #82
* Changes: Ping examples now will not show Dynamixel model number when communication is failed

3.5.1 (2017-08-18)
------------------
* Mac OS supports DynamixelSDK #51
* DynamixelSDK lib for Arduino (Arduino / OpenCR / OpenCM9.04) uploaded (TODO: Arduino Uno compatible DynamixelSDK light version)
* DynamixelSDK example for Arduino uploaded. It can be referred in OpenCR Repository (https://github.com/ROBOTIS-GIT/OpenCR/tree/master/arduino/opencr_arduino/opencr/libraries/OpenCR/examples/07.%20DynamixelSDK)
* DynamixelSDK LabVIEW can get communication result and Dynamixel error
* Standardizes folder structure of c, c++, ROS and Arduino c++ languages
* Fixes: Inconvenient way of getting meaning of packet result and error value #67
* Fixes: Misleading indentation warning in group_sync_read.c #91
* Fixes: Maximum length of port name is expanded to 100 #100
* Alternative: Include port_handler.h through relative path. #90
* Changes: Indent correction / Example tests & refresh / OS IFDEF
* Changes: Default Baudrate from 1000000(1M) bps to 57600 bps
* Changes: Macro for control table value changed to uints
* Changes: API references will be provided as doxygen (updates in c++ @ 3.5.1)
* Changes: License changed into Apache License .2.0 (Who are using SDK in previous license can use it as it is)
* Deprecated: printTxRxResult, printRxPacketError function will be unavailable in Dynamixel SDK 3.6.1

3.4.7 (2017-07-18)
------------------
* hotfix - Bug in Dynamixel group control is solved temporarily

3.4.6 (2017-07-07)
------------------
* hotfix - now DynamixelSDK for protocol1.0 supports read/write 4Byte (for XM series)

3.4.5 (2017-05-23)
------------------
* Merge branch 'kinetic-devel' of github.com:ROBOTIS-GIT/DynamixelSDK into kinetic-devel

3.4.4 (2017-04-26)
------------------
* hotfix - return delay time is changed from 4 into 8 due to the Ubuntu update 16.04.2

3.4.3 (2017-02-17)
------------------
* DynamixelSDK C++ ver. and ROS ver. in Windows platform now can use the port number of over then 10 #45

3.4.2 (2017-02-16)
------------------
* fprintf output in GroupBulkRead of C++ removed
* MATLAB library compiler error solving
* Makefile for build example sources in SBC added
* build files of windows c and c++ SDK rebuilt by using renewed SDK libraries
* example source of dxl_monitor - c and cpp ver modified #50
* Solved issue: #31, #34, #36, #50

3.4.1 (2016-08-22)
------------------
* Added ROS package folder for ROS users
* Modified c++'s original header files for ROS package

3.4.0 (2016-08-12)
------------------
* Added a ROS package information for ROS users

3.3.3 (2016-08-03)
------------------
* SDK C#     Resource Files comments Korean -> English
* SDK C#     properties comments Korean removed
* SDK C#     example default device path modified
* SDK All    License marks for example codes updated
* SDK Java   example source - folder name changed
* SDK MATLAB example code modified as platform version auto-detection #1
* SDK C/C++  build file for linux used by SBC(Single Board Computer)s updated #15
* Solved issue: #1, #15

3.3.2 (2016-06-30)
------------------
* SDK Python strange printout problem solved

3.3.1 (2016-06-30)
------------------
* SDK Python Errors in linux debugged

3.3.0 (2016-06-28)
------------------
* SDK C#      example as C version library binded source - released
* SDK Python  example as C version library binded source - released
* SDK Java    example as C version library binded source - released
* SDK MATLAB  example as C version library binded source - released
* SDK LabVIEW example as C version library binded source - released
* SDK C - Bug fixed (#8)
* Solved issue: #8

3.2.0 (2016-06-07)
------------------
* SDK C version - Code Refactoring
* SDK C version - Code style modified into ROS c++ code style

3.1.0 (2016-05-31)
------------------
* Code Refactoring
* Code style modified into ROS c++ code style
* License specified in the source code
* Solved issue: #3

3.0.3 (2016-05-18)
------------------
* Linux C version source codes uploaded

3.0.2 (2016-05-17)
------------------
* Windows C version source codes uploaded

3.0.1 (2016-04-26)
------------------
* Windows C++ version begun to be serviced

3.0.0 (2016-03-08)
------------------
* Linux C++ version source codes uploaded
