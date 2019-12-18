import socket
import asyncio
import websockets
import pathlib
import json

import util

async def consumer(message):
  await util.consumer_queue.put(json.loads(message))

async def producer():
  return await util.producer_queue.get()

async def consumer_handler(websocket, path):
  async for message in websocket:
    await consumer(message)

async def producer_handler(websocket, path):
  while True:
    message = await producer()
    await websocket.send(json.dumps(message))

async def handler(websocket, path):
  consumer_task = asyncio.ensure_future(
    consumer_handler(websocket, path))
  producer_task = asyncio.ensure_future(
    producer_handler(websocket, path))
  _, pending = await asyncio.wait(
    [consumer_task, producer_task],
    return_when=asyncio.FIRST_COMPLETED
  )
  for task in pending:
    task.cancel()

def start():
  start_server = websockets.serve(
    handler, "0.0.0.0", 5005
  )

  return start_server
