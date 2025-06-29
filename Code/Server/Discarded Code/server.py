# -*- coding: utf-8 -*-
"""
Created on Sat May 24 18:33:52 2025

@author: Joel Tapia Salvador
"""
import environment
import utils


class Server(environment.http.server.BaseHTTPRequestHandler):

    def log_request(self, code='No Code', size='No Size'):
        if isinstance(code, environment.http.server.HTTPStatus):
            code = code.value
        message = self.requestline
        utils.print_message(
            f'Request: {message} | Code: {code} | Size: {size}'
            + f' | From: {self.client_address[0]}:{self.client_address[1]}'
        )

    def log_error(self, message_format, *args):
        utils.print_error(
            error=(
                f'Error: {args[1]} | Code: {args[0]}'
                + f' | From: {self.client_address[0]}:{self.client_address[1]}'
            ),
            print_stack=False,
        )

    def log_message(self, format, *args):
        pass

    def _set_headers(self, content_type='text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def _serve_file(self, file_path, content_type):
        if environment.os.path.exists(file_path):
            self._set_headers(content_type)
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(500, f'File Not Found: "{file_path}"')

    def do_GET(self):
        parsed_path = environment.urllib.parse.unquote(self.path)
        if parsed_path == '/css/global_styles.css':
            self._serve_file(
                environment.os.path.join(
                    environment.PUBLIC_HTML_FILES_PATH,
                    'css',
                    'global_styles.css',
                ),
                'text/css',
            )

        elif parsed_path == '/images/background_photo.jpeg':
            self._serve_file(
                environment.os.path.join(
                    environment.PUBLIC_HTML_FILES_PATH,
                    'images',
                    'background_photo.jpeg',
                ),
                'image/jpeg',
            )

        elif parsed_path == '/':
            self._serve_file(
                environment.os.path.join(
                    environment.PUBLIC_HTML_FILES_PATH,
                    'main',
                    'placeholder.html',
                ),
                'text/html',
            )

        elif parsed_path == '/main/local_styles.css':
            self._serve_file(
                environment.os.path.join(
                    environment.PUBLIC_HTML_FILES_PATH,
                    'main',
                    'local_styles.css',
                ),
                'text/css',
            )

        else:
            self.send_error(404, f'Path "{self.path}" Not Found.')

    def do_POST(self):
        self.send_error(405, )


def run_server():
    http_server = environment.http.server.HTTPServer(
        (
            environment.SERVER_IP,
            environment.SERVER_PORT,
        ),
        Server,
    )
    utils.print_message(f'Starting server at {http_server.server_address}')

    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        utils.print_message('Server interrupted by user.')
    except Exception as error:
        utils.print_error(f'Unhandled server error: {error}')
    finally:
        http_server.server_close()
        utils.print_message('Server stopped.')


if __name__ == '__main__':
    try:
        environment.init()
        run_server()
    finally:
        environment.finish()
