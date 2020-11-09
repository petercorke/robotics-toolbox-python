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

from .robotis_def import *


class GroupBulkWrite:
    def __init__(self, port, ph):
        self.port = port
        self.ph = ph

        self.is_param_changed = False
        self.param = []
        self.data_list = {}

        self.clearParam()

    def makeParam(self):
        if self.ph.getProtocolVersion() == 1.0 or not self.data_list:
            return

        self.param = []

        for dxl_id in self.data_list:
            if not self.data_list[dxl_id]:
                return

            self.param.append(dxl_id)
            self.param.append(DXL_LOBYTE(self.data_list[dxl_id][1]))
            self.param.append(DXL_HIBYTE(self.data_list[dxl_id][1]))
            self.param.append(DXL_LOBYTE(self.data_list[dxl_id][2]))
            self.param.append(DXL_HIBYTE(self.data_list[dxl_id][2]))

            self.param.extend(self.data_list[dxl_id][0])

    def addParam(self, dxl_id, start_address, data_length, data):
        if self.ph.getProtocolVersion() == 1.0:
            return False

        if dxl_id in self.data_list:  # dxl_id already exist
            return False

        if len(data) > data_length:  # input data is longer than set
            return False

        self.data_list[dxl_id] = [data, start_address, data_length]

        self.is_param_changed = True
        return True

    def removeParam(self, dxl_id):
        if self.ph.getProtocolVersion() == 1.0:
            return

        if dxl_id not in self.data_list:  # NOT exist
            return

        del self.data_list[dxl_id]

        self.is_param_changed = True

    def changeParam(self, dxl_id, start_address, data_length, data):
        if self.ph.getProtocolVersion() == 1.0:
            return False

        if dxl_id not in self.data_list:  # NOT exist
            return False

        if len(data) > data_length:  # input data is longer than set
            return False

        self.data_list[dxl_id] = [data, start_address, data_length]

        self.is_param_changed = True
        return True

    def clearParam(self):
        if self.ph.getProtocolVersion() == 1.0:
            return

        self.data_list.clear()
        return

    def txPacket(self):
        if self.ph.getProtocolVersion() == 1.0 or len(self.data_list.keys()) == 0:
            return COMM_NOT_AVAILABLE

        if self.is_param_changed is True or len(self.param) == 0:
            self.makeParam()

        return self.ph.bulkWriteTxOnly(self.port, self.param, len(self.param))
