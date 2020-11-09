#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright 2017 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Author: Ryu Woon Jung (Leon)

from .protocol1_packet_handler import *
from .protocol2_packet_handler import *


def PacketHandler(protocol_version):
    # FIXME: float or int-to-float comparison can generate weird behaviour
    if protocol_version == 1.0:
        return Protocol1PacketHandler()
    elif protocol_version == 2.0:
        return Protocol2PacketHandler()
    else:
        return Protocol2PacketHandler()
