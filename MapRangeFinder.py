import cv2
import time
import keyboard
import numpy as np

import dxcam
from threading import Thread, Event
from PointOfInterestDetector import PointOfInterestDetectorDL

GAME_MAP_TOGGLE_KEY = 'm'
MANUALLY_ENABLE_KEY = '/'

SCALE_MINUS_KEY = '['
SCALE_PLUS_KEY = ']'

'''
RELATIVE_HEIGHT_MINUS_KEY = '-'
RELATIVE_HEIGHT_PLUS_KEY = '+'

RELATIVE_HEIGHT_MIN = -200
RELATIVE_HEIGHT_MAX = 200
HEIGHT_STEP = 10
'''

def draw_ref_box(image, scale):
    scales = {
        80: '1X: 8000M',
        38: '2X: 3800M',
        20: '4X: 2000M',
        10: '8X: 1000M',
        5: '16X: 500M'
    }
    height = image.shape[0]
    gap = int(height/scale)
    scale_hint = scales[scale]
    x0 = 50
    y0 = 50
    green = (0, 255, 0)
    image = cv2.rectangle(image, (x0, y0), (x0+gap, x0+gap), green, 2)
    image = cv2.putText(image, f'Scale: {scale_hint}', (int(x0/2), int(y0/2)), 0, 1, (255, 255, 0), 2)
    return image

def encode_webp(img, quality=85):
    return cv2.imencode('.webp', img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])[1]

def encode_jpeg(img, quality=90):
    return cv2.imencode('.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]

def decode(encoded_img):
    return cv2.imdecode(encoded_img, 1)

auto_resize = lambda img: cv2.resize(img, (int(img.shape[1]/img.shape[0]*1440), 1440)) if img.shape > (1600, 2560, 3) else img

import socket
import http.server
import socketserver

class ImageServer(http.server.SimpleHTTPRequestHandler):
    
    #override
    def log_message(self, format: str, *args) -> None:
        return True

    def do_GET(self):
        if self.path == '/':
            self.path = '/web.html'  # 确保路径正确无误
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.end_headers()
        self.wfile.write(self.server.image.tobytes()) 
        #self.wfile.write(json.dumps(self.server.image).encode())

class MapRangeFinder:

    scales = [80, 38, 20, 10, 5]

    def __init__(self, camera, poid, server, debugging = False, fps_cap = 10):
        self.camera = camera
        self.poid = poid
        self.server = server
        self.debugging = debugging
        self.fps_cap = fps_cap
        self.scale_id = 0
        self.scale = self.scales[self.scale_id]

        self.relative_height = 0

        self.toggled = False

        self.latest = None
        self.frame_grab_time = 0.0
        self.lines = []
        self.circles = []

        # Stats
        self.poi_time_costs = [0]

        # Events
        self.image_update_event = Event() # 更新屏幕截图
        self.poi_update_event = Event()   # 识别Point Of Interest
        self.drawing_event = Event()      # 根据结果绘图

    def toggle(self):
        self.toggled = not self.toggled
        print(f'Map Range Finder is {["OFF", "ON"][self.toggled]}')

    def toggle_on(self):
        self.toggled = True
        print(f'Map Range Finder is {["OFF", "ON"][self.toggled]}')

    def scale_up(self):
        if self.toggled:
            self.scale_id = min(self.scale_id + 1, len(self.scales)-1)
            self.scale = self.scales[self.scale_id]
            print(f'Scale is now {self.scale*100:6d} M')

    def scale_down(self):
        if self.toggled:
            self.scale_id = max(self.scale_id - 1, 0)
            self.scale = self.scales[self.scale_id]
            print(f'Scale is now {self.scale*100:6d} M')

    '''
    def relative_height_up(self):
        if self.toggled:
            self.relative_height = min(self.relative_height + HEIGHT_STEP, RELATIVE_HEIGHT_MAX)
            print(f'Relative height is now {self.relative_height} M')

    def relative_height_down(self):
        if self.toggled:
            self.relative_height = max(self.relative_height - HEIGHT_STEP, RELATIVE_HEIGHT_MIN)
            print(f'Relative height is now {self.relative_height} M')
    '''

    def get_frame(self):
        img = self.camera.get_latest_frame()
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = auto_resize(img)
        return img

    def start_threads(self):
        Thread(target=self.drawer, daemon=True).start()
        Thread(target=self.poi_update_thread, daemon=True).start()
        Thread(target=self.image_update_thread, daemon=True).start()
        Thread(target=self.update_manager_thread, daemon=True).start()

    def update_manager_thread(self):
        self.image_update_event.set()
        interval = 1/self.fps_cap
        while True:
            if self.toggled:
                self.image_update_event.set()
            time.sleep(interval)

    def image_update_thread(self):
        '''
        负责更新截图
        更新完成后，设置poi_update_event
        '''
        while True:
            self.image_update_event.wait()
            self.image_update_event.clear()
            if self.debugging:
                print('Start image update')

            updated = False
            start = time.time()
            while not updated:
                try:
                    #print('Trying to get frame')
                    self.latest = self.get_frame()
                    updated = True
                except Exception as e:
                    print(e)
            self.frame_grab_time = (time.time()-start)*1000
            if self.debugging:
                print(f'Image updated in {self.frame_grab_time:.2f} ms')

            self.poi_update_event.set() # 图像已更新，可以更新场景

    def poi_update_thread(self):
        while True:
            self.poi_update_event.wait()
            self.poi_update_event.clear()
            if self.debugging:
                print('Start poi update')
            start = time.time()
            try:
                lines, circles = poid.connect_player_mark(self.latest,
                                                          self.scale,
                                                          scene='map_l2plus',
                                                          relative_height=self.relative_height)
            except Exception as e:
                print(e)
                lines = []
                circles = []
            self.lines = lines
            self.circles = circles
            self.drawing_event.set()
            
            self.poi_time_costs.append(time.time()-start)
            self.poi_time_costs = self.poi_time_costs[-100:]

    def drawer(self):
        frametimes = []
        last_draw = time.time()
        draw_time = 0
        while True:
            self.drawing_event.wait()
            self.drawing_event.clear()
            
            # FPS
            frametime = time.time()-last_draw
            last_draw = time.time()
            frametimes.append(frametime)
            avg_frametime = np.average(frametimes[-10:])
            fps = 1/avg_frametime

            frame = self.latest
            
            # Map Scale
            scale = self.scale
            
            frame = draw_ref_box(frame, scale)

            # Stats
            poi_time = np.average(self.poi_time_costs[-5:])
            poi_lines_count = len(self.lines)
            poi_circles_count = len(self.circles)

            # Point of Interest
            if poi_lines_count > 0:
                frame = self.poid.draw_lines(frame, self.lines)
            if poi_circles_count > 0:
                frame = self.poid.draw_circles(frame, self.circles)
            
            # Stats text
            ## get current time in HH:MM:SS
            time_str = time.strftime("%H:%M:%S", time.localtime())
            stats_text_0 = f'[Update]  Time: {time_str} Grab Time: {self.frame_grab_time:.2f} ms'
            stats_text_1 = f'[Drawing] Time: {draw_time*1000:.2f} ms  FPS: {fps:.2f}  Frametime: {avg_frametime*1000:.2f} ms'
            stats_text_2 = f'[Scale]   Scale: {self.scale*100:6d} M  Relative Height: {self.relative_height:4d} M'
            stats_text_3 = f'[POI]     Time: {poi_time*1000:.2f} ms  Lines Count: {poi_lines_count}  Circles Count: {poi_circles_count}'
            
            if self.debugging:
                print(stats_text_0)
                print(stats_text_1)
                print(stats_text_2)
                print(stats_text_3)
            
            text_color = (0, 255, 0)
            frame = cv2.putText(frame, stats_text_0, (400, 40), 0, 1, text_color, 2)
            frame = cv2.putText(frame, stats_text_1, (400, 80), 0, 1, text_color, 2)
            frame = cv2.putText(frame, stats_text_2, (400, 120), 0, 1, text_color, 2)
            frame = cv2.putText(frame, stats_text_3, (400, 160), 0, 1, text_color, 2)

            # Update Image
            cv2.resize(frame, (1920, 1080))
            encoded_img = encode_jpeg(frame, 90)
            self.server.image = encoded_img #.tolist()
            draw_time = time.time() - last_draw
            if self.debugging:
                print('Drawing finished.')

if __name__ == '__main__':
    circle_model_path = 'circle_model.onnx'

    port = 8035

    target_fps = 4
    
    debugging = False
    
    poid = PointOfInterestDetectorDL(circle_model_path,
                                     is_onnx = True,
                                     min_distance = 100,
                                     max_distance = 9999,
                                     auto_crop = True,
                                     debugging = debugging
                                    )

    camera = dxcam.create(device_idx=0, output_idx=0)
    camera.start(target_fps=10)
    
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    encoded_img = encode_jpeg(dummy_img)

    server = socketserver.TCPServer(("::", port), ImageServer, bind_and_activate=False)
    server.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    server.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    server.server_bind()
    server.server_activate()

    server.image = encoded_img #.tolist()
    
    mrf = MapRangeFinder(camera, poid, server, debugging, target_fps)

    mrf.start_threads()

    keyboard.add_hotkey(GAME_MAP_TOGGLE_KEY, mrf.toggle)
    keyboard.add_hotkey(MANUALLY_ENABLE_KEY, mrf.toggle_on)
    keyboard.add_hotkey(SCALE_PLUS_KEY, mrf.scale_up)
    keyboard.add_hotkey(SCALE_MINUS_KEY, mrf.scale_down)
    '''
    keyboard.add_hotkey(RELATIVE_HEIGHT_PLUS_KEY, mrf.relative_height_up)
    keyboard.add_hotkey(RELATIVE_HEIGHT_MINUS_KEY, mrf.relative_height_down)
    '''
    
    server.serve_forever()