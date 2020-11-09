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


class GroupSyncRead:
    def __init__(self, port, ph, start_address, data_length):
        self.port = port
        self.ph = ph
        self.start_address = start_address
        self.data_length = data_length

        self.last_result = False
        self.is_param_changed = False
        self.param = []
        self.data_dict = {}

        self.clearParam()

    def makeParam(self):
        if self.ph.getProtocolVersion() == 1.0:
            return

        if not self.data_dict:  # len(self.data_dict.keys()) == 0:
            return

        self.param = []

        for dxl_id in self.data_dict:
            self.param.append(dxl_id)

    def addParam(self, dxl_id):
        if self.ph.getProtocolVersion() == 1.0:
            return False

        if dxl_id in self.data_dict:  # dxl_id already exist
            return False

        self.data_dict[dxl_id] = []  # [0] * self.data_length

        self.is_param_changed = True
        return True

    def removeParam(self, dxl_id):
        if self.ph.getProtocolVersion() == 1.0:
            return

        if dxl_id not in self.data_dict:  # NOT exist
            return

        del self.data_dict[dxl_id]

        self.is_param_changed = True

    def clearParam(self):
        if self.ph.getProtocolVersion() == 1.0:
            return

        self.data_dict.clear()

    def txPacket(self):
        if self.ph.getProtocolVersion() == 1.0 or len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE

        if self.is_param_changed is True or not self.param:
            self.makeParam()

        return self.ph.syncReadTx(self.port, self.start_address, self.data_length, self.param,
                                  len(self.data_dict.keys()) * 1)

    def rxPacket(self):
        self.last_result = False

        if self.ph.getProtocolVersion() == 1.0:
            return COMM_NOT_AVAILABLE

        result = COMM_RX_FAIL

        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE

        for dxl_id in self.data_dict:
            self.data_dict[dxl_id], result, _ = self.ph.readRx(self.port, dxl_id, self.data_length)
            if result != COMM_SUCCESS:
                return result

        if result == COMM_SUCCESS:
            self.last_result = True

        return result

    def txRxPacket(self):
        if self.ph.getProtocolVersion() == 1.0:
            return COMM_NOT_AVAILABLE

        result = self.txPacket()
        if result != COMM_SUCCESS:
            return result

        return self.rxPacket()

    def isAvailable(self, dxl_id, address, data_length):
        if self.ph.getProtocolVersion() == 1.0 or self.last_result is False or dxl_id not in self.data_dict:
            return False

        if (address < self.start_address) or (self.start_address + self.data_length - data_length < address):
            return False

        return True

    def getData(self, dxl_id, address, data_length):
        if not self.isAvailable(dxl_id, address, data_length):
            return 0

        if data_length == 1:
            return self.data_dict[dxl_id][address - self.start_address]
        elif data_length == 2:
            return DXL_MAKEWORD(self.data_dict[dxl_id][address - self.start_address],
                                self.data_dict[dxl_id][address - self.start_address + 1])
        elif data_length == 4:
            return DXL_MAKEDWORD(DXL_MAKEWORD(self.data_dict[dxl_id][address - self.start_address + 0],
                                              self.data_dict[dxl_id][address - self.start_address + 1]),
                                 DXL_MAKEWORD(self.data_dict[dxl_id][address - self.start_address + 2],
                                              self.data_dict[dxl_id][address - self.start_address + 3]))
        else:
            return 0
