import time
import numpy as np
import rerun as rr

rr.init("rerun_example_embed_web_viewer")
print("Rerun initialized")
# Now start the servers
url = rr.serve_grpc()
print(f"gRPC server started at: {url}")

# Generate and log the data first
positions = np.vstack([xyz.ravel() for xyz in np.mgrid[3 * [slice(-10, 10, 10j)]]]).T
colors = (
    np.vstack([rgb.ravel() for rgb in np.mgrid[3 * [slice(0, 255, 10j)]]])
    .astype(np.uint8)
    .T
)
rr.log("my_points", rr.Points3D(positions, colors=colors, radii=0.5))
print("Data logged")

web_port = 9999
rr.serve_web_viewer(web_port=web_port, connect_to=url)
print(f"Web viewer started at http://localhost:{web_port}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Ctrl-C received. Exiting.")
