import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import VehiclePlateRecognition

ROOT_DIR = f"{os.getcwd()}/website"


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, directory=ROOT_DIR)

    def do_POST(self):
        # Read the content length and the POST data from the request body
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        decoded_data = post_data.decode("utf-8")

        print(f"Received POST data: {decoded_data}")

        # Process the data (example: parse as JSON)
        try:
            json_data = json.loads(decoded_data)
            print(f"Parsed JSON: {json_data}")
        except json.JSONDecodeError:
            print("Data was not in valid JSON format.")
        result = VehiclePlateRecognition.recognizeLicensePlate()

        # Send the response back to the client
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))


def start(host: str, port: int):
    httpd = HTTPServer((host, port), RequestHandler)
    print(f"RequestHandler starts at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
