import os
import sys

import grpc

channel = grpc.insecure_channel(f"localhost:{os.getenv('GRPC_PORT', '50051')}")
try:
    grpc.channel_ready_future(channel).result(timeout=3)
except grpc.FutureTimeoutError:
    sys.exit(1)
