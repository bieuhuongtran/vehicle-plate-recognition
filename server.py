import traceback

import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import cgi

import VehiclePlateRecognition

ROOT_DIR = f"{os.getcwd()}/website"


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, directory=ROOT_DIR)

    def do_POST(self):
        try:
            ctype, pdict = cgi.parse_header(self.headers.get('Content-Type', ''))
            if ctype != 'multipart/form-data':
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status":"failed","message":"Cần multipart/form-data với field 'file'"}).encode("utf-8"))
                return

            content_len = int(self.headers.get('Content-Length', 0))
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            pdict['CONTENT-LENGTH'] = content_len
            fields = cgi.parse_multipart(self.rfile, pdict)

            if 'file' not in fields:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status":"failed","message":"Thiếu field 'file'"}).encode("utf-8"))
                return

            # Read the content length and the POST data from the request body
            imgBuffer = fields['file'][0]
            result = VehiclePlateRecognition.recognizeLicensePlate(imgBuffer)

            # Send the response back to the client
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status":"failed",
                "message": str(e),
                "trace": traceback.format_exc()
            }, ensure_ascii=False).encode("utf-8"))



def start(host: str, port: int):
    httpd = HTTPServer((host, port), RequestHandler)
    print(f"RequestHandler starts at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
