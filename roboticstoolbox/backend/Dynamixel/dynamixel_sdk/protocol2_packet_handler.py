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

TXPACKET_MAX_LEN = 1 * 1024
RXPACKET_MAX_LEN = 1 * 1024

# for Protocol 2.0 Packet
PKT_HEADER0 = 0
PKT_HEADER1 = 1
PKT_HEADER2 = 2
PKT_RESERVED = 3
PKT_ID = 4
PKT_LENGTH_L = 5
PKT_LENGTH_H = 6
PKT_INSTRUCTION = 7
PKT_ERROR = 8
PKT_PARAMETER0 = 8

# Protocol 2.0 Error bit
ERRNUM_RESULT_FAIL = 1  # Failed to process the instruction packet.
ERRNUM_INSTRUCTION = 2  # Instruction error
ERRNUM_CRC = 3  # CRC check error
ERRNUM_DATA_RANGE = 4  # Data range error
ERRNUM_DATA_LENGTH = 5  # Data length error
ERRNUM_DATA_LIMIT = 6  # Data limit error
ERRNUM_ACCESS = 7  # Access error

ERRBIT_ALERT = 128  # When the device has a problem, this bit is set to 1. Check "Device Status Check" value.


class Protocol2PacketHandler(object):
    def getProtocolVersion(self):
        return 2.0

    def getTxRxResult(self, result):
        if result == COMM_SUCCESS:
            return "[TxRxResult] Communication success!"
        elif result == COMM_PORT_BUSY:
            return "[TxRxResult] Port is in use!"
        elif result == COMM_TX_FAIL:
            return "[TxRxResult] Failed transmit instruction packet!"
        elif result == COMM_RX_FAIL:
            return "[TxRxResult] Failed get status packet from device!"
        elif result == COMM_TX_ERROR:
            return "[TxRxResult] Incorrect instruction packet!"
        elif result == COMM_RX_WAITING:
            return "[TxRxResult] Now receiving status packet!"
        elif result == COMM_RX_TIMEOUT:
            return "[TxRxResult] There is no status packet!"
        elif result == COMM_RX_CORRUPT:
            return "[TxRxResult] Incorrect status packet!"
        elif result == COMM_NOT_AVAILABLE:
            return "[TxRxResult] Protocol does not support this function!"
        else:
            return ""

    def getRxPacketError(self, error):
        if error & ERRBIT_ALERT:
            return "[RxPacketError] Hardware error occurred. Check the error at Control Table (Hardware Error Status)!"

        not_alert_error = error & ~ERRBIT_ALERT
        if not_alert_error == 0:
            return ""
        elif not_alert_error == ERRNUM_RESULT_FAIL:
            return "[RxPacketError] Failed to process the instruction packet!"

        elif not_alert_error == ERRNUM_INSTRUCTION:
            return "[RxPacketError] Undefined instruction or incorrect instruction!"

        elif not_alert_error == ERRNUM_CRC:
            return "[RxPacketError] CRC doesn't match!"

        elif not_alert_error == ERRNUM_DATA_RANGE:
            return "[RxPacketError] The data value is out of range!"

        elif not_alert_error == ERRNUM_DATA_LENGTH:
            return "[RxPacketError] The data length does not match as expected!"

        elif not_alert_error == ERRNUM_DATA_LIMIT:
            return "[RxPacketError] The data value exceeds the limit value!"

        elif not_alert_error == ERRNUM_ACCESS:
            return "[RxPacketError] Writing or Reading is not available to target address!"

        else:
            return "[RxPacketError] Unknown error code!"

    def updateCRC(self, crc_accum, data_blk_ptr, data_blk_size):
        crc_table = [0x0000,
                     0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
                     0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027,
                     0x0022, 0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D,
                     0x8077, 0x0072, 0x0050, 0x8055, 0x805F, 0x005A, 0x804B,
                     0x004E, 0x0044, 0x8041, 0x80C3, 0x00C6, 0x00CC, 0x80C9,
                     0x00D8, 0x80DD, 0x80D7, 0x00D2, 0x00F0, 0x80F5, 0x80FF,
                     0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1, 0x00A0, 0x80A5,
                     0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1, 0x8093,
                     0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
                     0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197,
                     0x0192, 0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE,
                     0x01A4, 0x81A1, 0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB,
                     0x01FE, 0x01F4, 0x81F1, 0x81D3, 0x01D6, 0x01DC, 0x81D9,
                     0x01C8, 0x81CD, 0x81C7, 0x01C2, 0x0140, 0x8145, 0x814F,
                     0x014A, 0x815B, 0x015E, 0x0154, 0x8151, 0x8173, 0x0176,
                     0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162, 0x8123,
                     0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
                     0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104,
                     0x8101, 0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D,
                     0x8317, 0x0312, 0x0330, 0x8335, 0x833F, 0x033A, 0x832B,
                     0x032E, 0x0324, 0x8321, 0x0360, 0x8365, 0x836F, 0x036A,
                     0x837B, 0x037E, 0x0374, 0x8371, 0x8353, 0x0356, 0x035C,
                     0x8359, 0x0348, 0x834D, 0x8347, 0x0342, 0x03C0, 0x83C5,
                     0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1, 0x83F3,
                     0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
                     0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7,
                     0x03B2, 0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E,
                     0x0384, 0x8381, 0x0280, 0x8285, 0x828F, 0x028A, 0x829B,
                     0x029E, 0x0294, 0x8291, 0x82B3, 0x02B6, 0x02BC, 0x82B9,
                     0x02A8, 0x82AD, 0x82A7, 0x02A2, 0x82E3, 0x02E6, 0x02EC,
                     0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2, 0x02D0, 0x82D5,
                     0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1, 0x8243,
                     0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
                     0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264,
                     0x8261, 0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E,
                     0x0234, 0x8231, 0x8213, 0x0216, 0x021C, 0x8219, 0x0208,
                     0x820D, 0x8207, 0x0202]

        for j in range(0, data_blk_size):
            i = ((crc_accum >> 8) ^ data_blk_ptr[j]) & 0xFF
            crc_accum = ((crc_accum << 8) ^ crc_table[i]) & 0xFFFF

        return crc_accum

    def addStuffing(self, packet):
        packet_length_in = DXL_MAKEWORD(packet[PKT_LENGTH_L], packet[PKT_LENGTH_H])
        packet_length_out = packet_length_in

        temp = [0] * TXPACKET_MAX_LEN

        # FF FF FD XX ID LEN_L LEN_H
        temp[PKT_HEADER0: PKT_HEADER0 + PKT_LENGTH_H + 1] = packet[PKT_HEADER0: PKT_HEADER0 + PKT_LENGTH_H + 1]

        index = PKT_INSTRUCTION

        for i in range(0, packet_length_in - 2):  # except CRC
            temp[index] = packet[i + PKT_INSTRUCTION]
            index = index + 1
            if packet[i + PKT_INSTRUCTION] == 0xFD \
                    and packet[i + PKT_INSTRUCTION - 1] == 0xFF \
                    and packet[i + PKT_INSTRUCTION - 2] == 0xFF:
                # FF FF FD
                temp[index] = 0xFD
                index = index + 1
                packet_length_out = packet_length_out + 1

        temp[index] = packet[PKT_INSTRUCTION + packet_length_in - 2]
        temp[index + 1] = packet[PKT_INSTRUCTION + packet_length_in - 1]
        index = index + 2

        if packet_length_in != packet_length_out:
            packet = [0] * index

        packet[0: index] = temp[0: index]

        packet[PKT_LENGTH_L] = DXL_LOBYTE(packet_length_out)
        packet[PKT_LENGTH_H] = DXL_HIBYTE(packet_length_out)

        return packet

    def removeStuffing(self, packet):
        packet_length_in = DXL_MAKEWORD(packet[PKT_LENGTH_L], packet[PKT_LENGTH_H])
        packet_length_out = packet_length_in

        index = PKT_INSTRUCTION
        for i in range(0, (packet_length_in - 2)):  # except CRC
            if (packet[i + PKT_INSTRUCTION] == 0xFD) and (packet[i + PKT_INSTRUCTION + 1] == 0xFD) and (
                    packet[i + PKT_INSTRUCTION - 1] == 0xFF) and (packet[i + PKT_INSTRUCTION - 2] == 0xFF):
                # FF FF FD FD
                packet_length_out = packet_length_out - 1
            else:
                packet[index] = packet[i + PKT_INSTRUCTION]
                index += 1

        packet[index] = packet[PKT_INSTRUCTION + packet_length_in - 2]
        packet[index + 1] = packet[PKT_INSTRUCTION + packet_length_in - 1]

        packet[PKT_LENGTH_L] = DXL_LOBYTE(packet_length_out)
        packet[PKT_LENGTH_H] = DXL_HIBYTE(packet_length_out)

        return packet

    def txPacket(self, port, txpacket):
        if port.is_using:
            return COMM_PORT_BUSY
        port.is_using = True

        # byte stuffing for header
        self.addStuffing(txpacket)

        # check max packet length
        total_packet_length = DXL_MAKEWORD(txpacket[PKT_LENGTH_L], txpacket[PKT_LENGTH_H]) + 7
        # 7: HEADER0 HEADER1 HEADER2 RESERVED ID LENGTH_L LENGTH_H

        if total_packet_length > TXPACKET_MAX_LEN:
            port.is_using = False
            return COMM_TX_ERROR

        # make packet header
        txpacket[PKT_HEADER0] = 0xFF
        txpacket[PKT_HEADER1] = 0xFF
        txpacket[PKT_HEADER2] = 0xFD
        txpacket[PKT_RESERVED] = 0x00

        # add CRC16
        crc = self.updateCRC(0, txpacket, total_packet_length - 2)  # 2: CRC16

        txpacket[total_packet_length - 2] = DXL_LOBYTE(crc)
        txpacket[total_packet_length - 1] = DXL_HIBYTE(crc)

        # tx packet
        port.clearPort()
        written_packet_length = port.writePort(txpacket)
        if total_packet_length != written_packet_length:
            port.is_using = False
            return COMM_TX_FAIL

        return COMM_SUCCESS

    def rxPacket(self, port):
        rxpacket = []

        result = COMM_TX_FAIL
        rx_length = 0
        wait_length = 11  # minimum length (HEADER0 HEADER1 HEADER2 RESERVED ID LENGTH_L LENGTH_H INST ERROR CRC16_L CRC16_H)

        while True:
            rxpacket.extend(port.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                # find packet header
                for idx in range(0, (rx_length - 3)):
                    if (rxpacket[idx] == 0xFF) and (rxpacket[idx + 1] == 0xFF) and (rxpacket[idx + 2] == 0xFD) and (
                            rxpacket[idx + 3] != 0xFD):
                        break

                if idx == 0:
                    if (rxpacket[PKT_RESERVED] != 0x00) or (rxpacket[PKT_ID] > 0xFC) or (
                            DXL_MAKEWORD(rxpacket[PKT_LENGTH_L], rxpacket[PKT_LENGTH_H]) > RXPACKET_MAX_LEN) or (
                            rxpacket[PKT_INSTRUCTION] != 0x55):
                        # remove the first byte in the packet
                        del rxpacket[0]
                        rx_length -= 1
                        continue

                    if wait_length != (DXL_MAKEWORD(rxpacket[PKT_LENGTH_L], rxpacket[PKT_LENGTH_H]) + PKT_LENGTH_H + 1):
                        wait_length = DXL_MAKEWORD(rxpacket[PKT_LENGTH_L], rxpacket[PKT_LENGTH_H]) + PKT_LENGTH_H + 1
                        continue

                    if rx_length < wait_length:
                        if port.isPacketTimeout():
                            if rx_length == 0:
                                result = COMM_RX_TIMEOUT
                            else:
                                result = COMM_RX_CORRUPT
                            break
                        else:
                            continue

                    crc = DXL_MAKEWORD(rxpacket[wait_length - 2], rxpacket[wait_length - 1])

                    if self.updateCRC(0, rxpacket, wait_length - 2) == crc:
                        result = COMM_SUCCESS
                    else:
                        result = COMM_RX_CORRUPT
                    break

                else:
                    # remove unnecessary packets
                    del rxpacket[0: idx]
                    rx_length -= idx

            else:
                if port.isPacketTimeout():
                    if rx_length == 0:
                        result = COMM_RX_TIMEOUT
                    else:
                        result = COMM_RX_CORRUPT
                    break

        port.is_using = False

        if result == COMM_SUCCESS:
            rxpacket = self.removeStuffing(rxpacket)

        return rxpacket, result

    # NOT for BulkRead / SyncRead instruction
    def txRxPacket(self, port, txpacket):
        rxpacket = None
        error = 0

        # tx packet
        result = self.txPacket(port, txpacket)
        if result != COMM_SUCCESS:
            return rxpacket, result, error

        # (Instruction == BulkRead or SyncRead) == this function is not available.
        if txpacket[PKT_INSTRUCTION] == INST_BULK_READ or txpacket[PKT_INSTRUCTION] == INST_SYNC_READ:
            result = COMM_NOT_AVAILABLE

        # (ID == Broadcast ID) == no need to wait for status packet or not available.
        # (Instruction == action) == no need to wait for status packet
        if txpacket[PKT_ID] == BROADCAST_ID or txpacket[PKT_INSTRUCTION] == INST_ACTION:
            port.is_using = False
            return rxpacket, result, error

        # set packet timeout
        if txpacket[PKT_INSTRUCTION] == INST_READ:
            port.setPacketTimeout(DXL_MAKEWORD(txpacket[PKT_PARAMETER0 + 2], txpacket[PKT_PARAMETER0 + 3]) + 11)
        else:
            port.setPacketTimeout(11)
            # HEADER0 HEADER1 HEADER2 RESERVED ID LENGTH_L LENGTH_H INST ERROR CRC16_L CRC16_H

        # rx packet
        while True:
            rxpacket, result = self.rxPacket(port)
            if result != COMM_SUCCESS or txpacket[PKT_ID] == rxpacket[PKT_ID]:
                break

        if result == COMM_SUCCESS and txpacket[PKT_ID] == rxpacket[PKT_ID]:
            error = rxpacket[PKT_ERROR]

        return rxpacket, result, error

    def ping(self, port, dxl_id):
        model_number = 0
        error = 0

        txpacket = [0] * 10

        if dxl_id >= BROADCAST_ID:
            return model_number, COMM_NOT_AVAILABLE, error

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = 3
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_PING

        rxpacket, result, error = self.txRxPacket(port, txpacket)
        if result == COMM_SUCCESS:
            model_number = DXL_MAKEWORD(rxpacket[PKT_PARAMETER0 + 1], rxpacket[PKT_PARAMETER0 + 2])

        return model_number, result, error

    def broadcastPing(self, port):
        data_list = {}

        STATUS_LENGTH = 14

        rx_length = 0
        wait_length = STATUS_LENGTH * MAX_ID

        txpacket = [0] * 10
        rxpacket = []

        tx_time_per_byte = (1000.0 / port.getBaudRate()) *10.0;

        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH_L] = 3
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_PING

        result = self.txPacket(port, txpacket)
        if result != COMM_SUCCESS:
            port.is_using = False
            return data_list, result

        # set rx timeout
        #port.setPacketTimeout(wait_length * 1)
        port.setPacketTimeoutMillis((wait_length * tx_time_per_byte) + (3.0 * MAX_ID) + 16.0);

        while True:
            rxpacket += port.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)

            if port.isPacketTimeout():  # or rx_length >= wait_length
                break

        port.is_using = False

        if rx_length == 0:
            return data_list, COMM_RX_TIMEOUT

        while True:
            if rx_length < STATUS_LENGTH:
                return data_list, COMM_RX_CORRUPT

            # find packet header
            for idx in range(0, rx_length - 2):
                if rxpacket[idx] == 0xFF and rxpacket[idx + 1] == 0xFF and rxpacket[idx + 2] == 0xFD:
                    break

            if idx == 0:  # found at the beginning of the packet
                # verify CRC16
                crc = DXL_MAKEWORD(rxpacket[STATUS_LENGTH - 2], rxpacket[STATUS_LENGTH - 1])

                if self.updateCRC(0, rxpacket, STATUS_LENGTH - 2) == crc:
                    result = COMM_SUCCESS

                    data_list[rxpacket[PKT_ID]] = [
                        DXL_MAKEWORD(rxpacket[PKT_PARAMETER0 + 1], rxpacket[PKT_PARAMETER0 + 2]),
                        rxpacket[PKT_PARAMETER0 + 3]]

                    del rxpacket[0: STATUS_LENGTH]
                    rx_length = rx_length - STATUS_LENGTH

                    if rx_length == 0:
                        return data_list, result

                else:
                    result = COMM_RX_CORRUPT

                    # remove header (0xFF 0xFF 0xFD)
                    del rxpacket[0: 3]
                    rx_length = rx_length - 3

            else:
                # remove unnecessary packets
                del rxpacket[0: idx]
                rx_length = rx_length - idx

        # FIXME: unreachable code
        return data_list, result

    def action(self, port, dxl_id):
        txpacket = [0] * 10

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = 3
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_ACTION

        _, result, _ = self.txRxPacket(port, txpacket)
        return result

    def reboot(self, port, dxl_id):
        txpacket = [0] * 10

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = 3
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_REBOOT

        _, result, error = self.txRxPacket(port, txpacket)
        return result, error

    def clearMultiTurn(self, port, dxl_id):
        txpacket = [0] * 15

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = 8
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_CLEAR
        txpacket[PKT_PARAMETER0 + 0] = 0x01
        txpacket[PKT_PARAMETER0 + 1] = 0x44
        txpacket[PKT_PARAMETER0 + 2] = 0x58
        txpacket[PKT_PARAMETER0 + 3] = 0x4C
        txpacket[PKT_PARAMETER0 + 4] = 0x22

        _, result, error = self.txRxPacket(port, txpacket)
        return result, error

    def factoryReset(self, port, dxl_id, option):
        txpacket = [0] * 11

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = 4
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_FACTORY_RESET
        txpacket[PKT_PARAMETER0] = option

        _, result, error = self.txRxPacket(port, txpacket)
        return result, error

    def readTx(self, port, dxl_id, address, length):
        txpacket = [0] * 14

        if dxl_id >= BROADCAST_ID:
            return COMM_NOT_AVAILABLE

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = 7
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_READ
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(address)
        txpacket[PKT_PARAMETER0 + 2] = DXL_LOBYTE(length)
        txpacket[PKT_PARAMETER0 + 3] = DXL_HIBYTE(length)

        result = self.txPacket(port, txpacket)

        # set packet timeout
        if result == COMM_SUCCESS:
            port.setPacketTimeout(length + 11)

        return result

    def readRx(self, port, dxl_id, length):
        result = COMM_TX_FAIL
        error = 0

        rxpacket = None
        data = []

        while True:
            rxpacket, result = self.rxPacket(port)

            if result != COMM_SUCCESS or rxpacket[PKT_ID] == dxl_id:
                break

        if result == COMM_SUCCESS and rxpacket[PKT_ID] == dxl_id:
            error = rxpacket[PKT_ERROR]

            data.extend(rxpacket[PKT_PARAMETER0 + 1: PKT_PARAMETER0 + 1 + length])

        return data, result, error

    def readTxRx(self, port, dxl_id, address, length):
        error = 0

        txpacket = [0] * 14
        data = []

        if dxl_id >= BROADCAST_ID:
            return data, COMM_NOT_AVAILABLE, error

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = 7
        txpacket[PKT_LENGTH_H] = 0
        txpacket[PKT_INSTRUCTION] = INST_READ
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(address)
        txpacket[PKT_PARAMETER0 + 2] = DXL_LOBYTE(length)
        txpacket[PKT_PARAMETER0 + 3] = DXL_HIBYTE(length)

        rxpacket, result, error = self.txRxPacket(port, txpacket)
        if result == COMM_SUCCESS:
            error = rxpacket[PKT_ERROR]

            data.extend(rxpacket[PKT_PARAMETER0 + 1: PKT_PARAMETER0 + 1 + length])

        return data, result, error

    def read1ByteTx(self, port, dxl_id, address):
        return self.readTx(port, dxl_id, address, 1)

    def read1ByteRx(self, port, dxl_id):
        data, result, error = self.readRx(port, dxl_id, 1)
        data_read = data[0] if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def read1ByteTxRx(self, port, dxl_id, address):
        data, result, error = self.readTxRx(port, dxl_id, address, 1)
        data_read = data[0] if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def read2ByteTx(self, port, dxl_id, address):
        return self.readTx(port, dxl_id, address, 2)

    def read2ByteRx(self, port, dxl_id):
        data, result, error = self.readRx(port, dxl_id, 2)
        data_read = DXL_MAKEWORD(data[0], data[1]) if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def read2ByteTxRx(self, port, dxl_id, address):
        data, result, error = self.readTxRx(port, dxl_id, address, 2)
        data_read = DXL_MAKEWORD(data[0], data[1]) if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def read4ByteTx(self, port, dxl_id, address):
        return self.readTx(port, dxl_id, address, 4)

    def read4ByteRx(self, port, dxl_id):
        data, result, error = self.readRx(port, dxl_id, 4)
        data_read = DXL_MAKEDWORD(DXL_MAKEWORD(data[0], data[1]),
                                  DXL_MAKEWORD(data[2], data[3])) if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def read4ByteTxRx(self, port, dxl_id, address):
        data, result, error = self.readTxRx(port, dxl_id, address, 4)
        data_read = DXL_MAKEDWORD(DXL_MAKEWORD(data[0], data[1]),
                                  DXL_MAKEWORD(data[2], data[3])) if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def writeTxOnly(self, port, dxl_id, address, length, data):
        txpacket = [0] * (length + 12)

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(length + 5)
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(length + 5)
        txpacket[PKT_INSTRUCTION] = INST_WRITE
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(address)

        txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + length] = data[0: length]

        result = self.txPacket(port, txpacket)
        port.is_using = False

        return result

    def writeTxRx(self, port, dxl_id, address, length, data):
        txpacket = [0] * (length + 12)

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(length + 5)
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(length + 5)
        txpacket[PKT_INSTRUCTION] = INST_WRITE
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(address)

        txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + length] = data[0: length]
        rxpacket, result, error = self.txRxPacket(port, txpacket)

        return result, error

    def write1ByteTxOnly(self, port, dxl_id, address, data):
        data_write = [data]
        return self.writeTxOnly(port, dxl_id, address, 1, data_write)

    def write1ByteTxRx(self, port, dxl_id, address, data):
        data_write = [data]
        return self.writeTxRx(port, dxl_id, address, 1, data_write)

    def write2ByteTxOnly(self, port, dxl_id, address, data):
        data_write = [DXL_LOBYTE(data), DXL_HIBYTE(data)]
        return self.writeTxOnly(port, dxl_id, address, 2, data_write)

    def write2ByteTxRx(self, port, dxl_id, address, data):
        data_write = [DXL_LOBYTE(data), DXL_HIBYTE(data)]
        return self.writeTxRx(port, dxl_id, address, 2, data_write)

    def write4ByteTxOnly(self, port, dxl_id, address, data):
        data_write = [DXL_LOBYTE(DXL_LOWORD(data)),
                      DXL_HIBYTE(DXL_LOWORD(data)),
                      DXL_LOBYTE(DXL_HIWORD(data)),
                      DXL_HIBYTE(DXL_HIWORD(data))]
        return self.writeTxOnly(port, dxl_id, address, 4, data_write)

    def write4ByteTxRx(self, port, dxl_id, address, data):
        data_write = [DXL_LOBYTE(DXL_LOWORD(data)),
                      DXL_HIBYTE(DXL_LOWORD(data)),
                      DXL_LOBYTE(DXL_HIWORD(data)),
                      DXL_HIBYTE(DXL_HIWORD(data))]
        return self.writeTxRx(port, dxl_id, address, 4, data_write)

    def regWriteTxOnly(self, port, dxl_id, address, length, data):
        txpacket = [0] * (length + 12)

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(length + 5)
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(length + 5)
        txpacket[PKT_INSTRUCTION] = INST_REG_WRITE
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(address)

        txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + length] = data[0: length]

        result = self.txPacket(port, txpacket)
        port.is_using = False

        return result

    def regWriteTxRx(self, port, dxl_id, address, length, data):
        txpacket = [0] * (length + 12)

        txpacket[PKT_ID] = dxl_id
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(length + 5)
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(length + 5)
        txpacket[PKT_INSTRUCTION] = INST_REG_WRITE
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(address)

        txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + length] = data[0: length]

        _, result, error = self.txRxPacket(port, txpacket)

        return result, error

    def syncReadTx(self, port, start_address, data_length, param, param_length):
        txpacket = [0] * (param_length + 14)
        # 14: HEADER0 HEADER1 HEADER2 RESERVED ID LEN_L LEN_H INST START_ADDR_L START_ADDR_H DATA_LEN_L DATA_LEN_H CRC16_L CRC16_H

        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(
            param_length + 7)  # 7: INST START_ADDR_L START_ADDR_H DATA_LEN_L DATA_LEN_H CRC16_L CRC16_H
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(
            param_length + 7)  # 7: INST START_ADDR_L START_ADDR_H DATA_LEN_L DATA_LEN_H CRC16_L CRC16_H
        txpacket[PKT_INSTRUCTION] = INST_SYNC_READ
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(start_address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(start_address)
        txpacket[PKT_PARAMETER0 + 2] = DXL_LOBYTE(data_length)
        txpacket[PKT_PARAMETER0 + 3] = DXL_HIBYTE(data_length)

        txpacket[PKT_PARAMETER0 + 4: PKT_PARAMETER0 + 4 + param_length] = param[0: param_length]

        result = self.txPacket(port, txpacket)
        if result == COMM_SUCCESS:
            port.setPacketTimeout((11 + data_length) * param_length)

        return result

    def syncWriteTxOnly(self, port, start_address, data_length, param, param_length):
        txpacket = [0] * (param_length + 14)
        # 14: HEADER0 HEADER1 HEADER2 RESERVED ID LEN_L LEN_H INST START_ADDR_L START_ADDR_H DATA_LEN_L DATA_LEN_H CRC16_L CRC16_H

        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(
            param_length + 7)  # 7: INST START_ADDR_L START_ADDR_H DATA_LEN_L DATA_LEN_H CRC16_L CRC16_H
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(
            param_length + 7)  # 7: INST START_ADDR_L START_ADDR_H DATA_LEN_L DATA_LEN_H CRC16_L CRC16_H
        txpacket[PKT_INSTRUCTION] = INST_SYNC_WRITE
        txpacket[PKT_PARAMETER0 + 0] = DXL_LOBYTE(start_address)
        txpacket[PKT_PARAMETER0 + 1] = DXL_HIBYTE(start_address)
        txpacket[PKT_PARAMETER0 + 2] = DXL_LOBYTE(data_length)
        txpacket[PKT_PARAMETER0 + 3] = DXL_HIBYTE(data_length)

        txpacket[PKT_PARAMETER0 + 4: PKT_PARAMETER0 + 4 + param_length] = param[0: param_length]

        _, result, _ = self.txRxPacket(port, txpacket)

        return result

    def bulkReadTx(self, port, param, param_length):
        txpacket = [0] * (param_length + 10)
        # 10: HEADER0 HEADER1 HEADER2 RESERVED ID LEN_L LEN_H INST CRC16_L CRC16_H

        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(param_length + 3)  # 3: INST CRC16_L CRC16_H
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(param_length + 3)  # 3: INST CRC16_L CRC16_H
        txpacket[PKT_INSTRUCTION] = INST_BULK_READ

        txpacket[PKT_PARAMETER0: PKT_PARAMETER0 + param_length] = param[0: param_length]

        result = self.txPacket(port, txpacket)
        if result == COMM_SUCCESS:
            wait_length = 0
            i = 0
            while i < param_length:
                wait_length += DXL_MAKEWORD(param[i + 3], param[i + 4]) + 10
                i += 5
            port.setPacketTimeout(wait_length)

        return result

    def bulkWriteTxOnly(self, port, param, param_length):
        txpacket = [0] * (param_length + 10)
        # 10: HEADER0 HEADER1 HEADER2 RESERVED ID LEN_L LEN_H INST CRC16_L CRC16_H

        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH_L] = DXL_LOBYTE(param_length + 3)  # 3: INST CRC16_L CRC16_H
        txpacket[PKT_LENGTH_H] = DXL_HIBYTE(param_length + 3)  # 3: INST CRC16_L CRC16_H
        txpacket[PKT_INSTRUCTION] = INST_BULK_WRITE

        txpacket[PKT_PARAMETER0: PKT_PARAMETER0 + param_length] = param[0: param_length]

        _, result, _ = self.txRxPacket(port, txpacket)

        return result
