
#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp

# # Launch the simulator Swift
# env = rtb.backend.Swift()
# env.launch()

# Create a Panda robot object
panda = rtb.models.Panda()

# print(panda)
# print(panda.base_link)
# print(panda.ee_links)

# path, n = panda.get_path(panda.base_link, panda.ee_links[0])

# q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
# panda.q = q1

# print(panda.fkine())

# for link in path:
#     print(link.name)

# print(panda.get_path(panda.base_link, panda.ee_links[0])[0])

# print(panda.links[5].A(0))

# # Set joint angles to ready configuration
# panda.q = panda.qr

# Add the Panda to the simulator
# env.add(panda)


# while 1:
#     pass










# import webbrowser as wb
# import asyncio
# import datetime
# import random
# import websockets
# import threading
# import time
# from queue import Queue


# class Socket:

#     def __init__(self):
#         print('Server Started')
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)

#         start_server = websockets.serve(self.serve, "localhost", 8997)

#         self.loop.run_until_complete(start_server)
#         self.loop.run_forever()

#     async def serve(self, websocket, path):
#         # while True:
#             # message = await self.producer()
#         await websocket.send('51111')
#         self.loop.stop()

#     # async def producer(self):
#     #     data = self.q.get()
#     #     # await asyncio.sleep(1)
#     #     # now = datetime.datetime.utcnow().isoformat() + "Z"

#     #     return data


# # q = Queue()
# # # x = threading.Thread(target=Socket, args=(q, ), daemon=True)
# # # x.start()
# Socket()


# while 1:
#     time.sleep(1)
#     print('hello')
#     # q.put('hi')


# wb.open_new_tab('http://www.google.com')

# def send(data):



# panda = rp.models.Panda()
# panda.q = panda.qr

# Tep = panda.fkine() * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.2)

# arrived = False
# # env.add(panda)

# dt = 0.05

# while not arrived:

#     v, arrived = rp.p_servo(panda.fkine(), Tep, 1)
#     panda.qd = np.linalg.pinv(panda.jacobe()) @ v
#     time.sleep(1)

    # env.step(50)
