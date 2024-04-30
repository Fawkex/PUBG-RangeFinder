import cv2
import math
import numpy as np
import multiprocessing as mp

def crop_center(image):
    height, width = image.shape[0:2]
    target_height = int(height*0.8)
    w = height
    x = width/2 - w/2
    offset_x = x
    offset_y = int(height*0.1)
    return image[offset_y: offset_y+target_height, int(x): int(x+w)], offset_x, offset_y

def crop_by_label(image, label):
    assert(label in ['minimap_l1', 'minimap_l2', 'map_l1', 'map_l2plus'])
    height, width = image.shape[0:2]
    if label == 'map_l2plus':
        # 去除左右各20%，只取中间60%
        left = int(width*0.2)
        right = int(width*0.8)
        # 去除上下各5%，只取中间90%
        up = int(height*0.05)
        down = int(height*0.95)
        offset_x = left
        offset_y = up
        return image[up: down, left: right], offset_x, offset_y
    if label == 'map_l1':
        w = height
        x = width/2 - w/2
        offset_x = x
        offset_y = 0
        return image[0: height, int(x): int(x+w)], offset_x, offset_y
    if label == 'minimap_l1':
        offset_x = int(width*0.8484375)
        offset_y = int(height*0.73333333333)
        map_height = map_width = int(height*0.2375)
        return image[offset_y: offset_y+map_height, offset_x: offset_x+map_width], offset_x, offset_y
    if label == 'minimap_l2':
        offset_x = int(width*0.745)
        offset_y = int(height*0.55)
        map_height = map_width = int(height*0.423)
        return image[offset_y: offset_y+map_height, offset_x: offset_x+map_width], offset_x, offset_y

class PointOfInterestDetector:
    '''
    Circle Finding Method
    '''
    def __init__(self, player_min_dist = 25,
                 player_param1 = 80,
                 player_param2 = 25,
                 player_min_rad_ratio = 0.0065,
                 player_max_rad_ratio = 0.0112,
                 mark_min_dist = 50,
                 mark_param1 = 80,
                 mark_param2 = 20,
                 mark_min_rad_ratio = 0.005,
                 mark_max_rad_ratio = 0.0065,
                 min_distance = 100, # in metres
                 max_distance = 2000, 
                 crop_to_center = False,
                 multiprocessing = True
                ):
        
        self.player_min_dist = player_min_dist
        self.player_param1 = player_param1
        self.player_param2 = player_param2
        self.player_min_rad_ratio = player_min_rad_ratio
        self.player_max_rad_ratio = player_max_rad_ratio
        self.mark_min_dist = mark_min_dist
        self.mark_param1 = mark_param1
        self.mark_param2 = mark_param2
        self.mark_min_rad_ratio = mark_min_rad_ratio
        self.mark_max_rad_ratio = mark_max_rad_ratio
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.crop_to_center = crop_to_center
        self.multiprocessing = multiprocessing
        if multiprocessing:
            self.pool = mp.Pool(2)

    def get_player_pos(self, image):
        img = image.copy()
        if self.crop_to_center:
            img, of_x, of_y = crop_center(img)
        height = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_rad = int(self.player_min_rad_ratio*height)
        max_rad = int(self.player_max_rad_ratio*height)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, self.player_min_dist,
                            param1=self.player_param1, param2=self.player_param2,
                            minRadius=min_rad, maxRadius=max_rad)
        if circles is None:
            return []
        ret = []
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, radius = i[0:3]
            if self.crop_to_center:
                x += of_x
                y += of_y
            ret.append([int(x), int(y), radius])
        return ret
        
    def get_mark_pos(self, image):
        img = image.copy()
        if self.crop_to_center:
            img, of_x, of_y = crop_center(img)
        height = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_rad = int(self.mark_min_rad_ratio*height)
        max_rad = int(self.mark_max_rad_ratio*height)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, self.mark_min_dist,
                            param1=self.mark_param1, param2=self.mark_param2,
                            minRadius=min_rad, maxRadius=max_rad)
        if circles is None:
            return []
        ret = []
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, radius = i[0:3]
            if self.crop_to_center:
                x += of_x
                y += of_y
            ret.append([int(x), int(y), radius])
        return ret

    def connect_player_mark(self, image, scale):
        cal_dis = lambda x1,y1,x2,y2:math.sqrt(abs(x1-x2)**2+abs(y1-y2)**2)
        avg_color = lambda a,b:(int(a[0]*0.5+b[0]*0.5), int(a[1]*0.5+b[1]*0.5), int(a[2]*0.5+b[2]*0.5))

        height = image.shape[0]
        pix_per_100m = height/scale

        if self.multiprocessing:
            players_result = self.pool.apply_async(self.get_player_pos, (image, ))
            marks_result = self.pool.apply_async(self.get_mark_pos, (image, ))
            players = players_result.get()
            marks = marks_result.get()
        else:
            players = self.get_player_pos(image)
            marks = self.get_mark_pos(image)

        rad_player = int(height*0.0105)
        rad_mark = int(height*0.009)
        
        lines = [] # [[x1, y1, x2, y2, color, distance],...]
        circles = [ [m[0], m[1], rad_mark, (255,255,153)] for m in marks ] # [[x, y, rad, color, thick]]
        
        colors = [
            (51,255,255), # Yellow 
            (51,255,51),   # Green
            (204,102,0), # Blue
            (102,0,204), # Wine
            (204,204,0), # Cyan
            (255,0,127)  # Purple 
        ]
        color_id = 0

        distincted_marks = []
        for mark in marks:
            m_x, m_y, _ = mark
            for player in players:
                p_x, p_y, _ = player
                distance_pix = cal_dis(p_x, p_y, m_x, m_y)
                if (distance_pix) > 30:
                    distincted_marks.append(mark)

        marks = distincted_marks
        
        for i, player in enumerate(players):
            p_x, p_y, _ = player
            color = colors[color_id]
            color_id = (color_id+1)%6
            circle = [p_x, p_y, rad_player, color]
            circles.append(circle)
            for mark in marks:
                m_x, m_y, m_rad = mark
                distance_pix = cal_dis(p_x, p_y, m_x, m_y)
                distance_m = int(distance_pix/pix_per_100m*100)
                if (distance_m > self.min_distance and distance_m < self.max_distance):
                    line = [p_x, p_y, m_x, m_y, color, distance_m]
                    lines.append(line)
            for player2 in players[i+1:]:
                p2_x, p2_y, _ = player2
                p2p_color = avg_color(color, (255,255,255))
                distance_pix = cal_dis(p_x, p_y, p2_x, p2_y)
                distance_m = int(distance_pix/pix_per_100m*100)
                if (distance_m > self.min_distance and distance_m < self.max_distance):
                    line = [p_x, p_y, p2_x, p2_y, p2p_color, distance_m]
                    lines.append(line)
                
        return lines, circles

    def draw_lines(self, image, lines):
        img = image.copy()
        for line in lines:
            start_point = (line[0], line[1])
            end_point = (line[2], line[3])
            color = line[4]
            distance = f'{line[5]}'
            offset = 25
            if (start_point[0] > end_point[0]):
                offset = -25
            cv2.line(img, start_point, end_point, color, 2)
            mid_point = (int(start_point[0]*0.5+end_point[0]*0.5), int(start_point[1]*0.5+end_point[1]*0.5))
            cv2.putText(img, distance, mid_point, 0, 1, color, 2)
        return img

    def draw_circles(self, image, circles):
        img = image.copy()
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            color = circle[3]
            img = cv2.circle(img, center, radius, color, 3)
        return img

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

from CircleDetector import CircleDetector

class PointOfInterestDetectorDL:
    '''
    Circle Finding Method
    '''
    def __init__(self, model_path,
                 is_onnx = False,
                 min_dist = 0.016,
                 param1 = 80,
                 param2 = 22,
                 min_rad_ratio = 0.006,
                 max_rad_ratio = 0.012,
                 min_distance = 100, # in metres
                 max_distance = 3000, 
                 auto_crop = False,
                 debugging = False
                ):
        
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_rad_ratio = min_rad_ratio
        self.max_rad_ratio = max_rad_ratio
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.auto_crop = auto_crop
        self.debugging = debugging
        self.cd = CircleDetector(model_path, is_onnx = is_onnx)

    def get_all_circle_pos(self, image, scene='map_l2plus'):
        img = image.copy()
        height = img.shape[0]
        if self.auto_crop:
            img, off_x, off_y = crop_by_label(img, scene)
        else:
            off_x = off_y = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_rad = int(self.min_rad_ratio*height)
        max_rad = int(self.max_rad_ratio*height)
        if self.min_dist < 1:
            min_dist = int(self.min_dist * height)
        else:
            min_dist = self.min_dist
        #print(min_dist, self.param1, self.param2, min_rad, max_rad)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_dist,
                            param1=self.param1, param2=self.param2,
                            minRadius=min_rad, maxRadius=max_rad)
        if circles is None:
            return []
        ret = []
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, radius = i[0:3]
            x += off_x
            y += off_y
            ret.append([int(x), int(y), radius])
        return ret
    
    def get_all_circles(self, image, scene='map_l2plus'):
        circles = self.get_all_circle_pos(image, scene)
        height = image.shape[0]
        rad = int(height * 0.0125)
        images = []
        if self.debugging:
            print(f'Detected {len(circles)} circles.')
        for circle in circles:
            x, y, _ = circle
            left = x - rad
            right = x + rad
            down = y - rad
            up = y + rad
            cropped_image = image[down: up, left: right]
            images.append(cropped_image)
        return circles, images

    def get_pos(self, image, scene='map_l2plus'):
        circles, circles_images = self.get_all_circles(image, scene)
        predicted_labels = self.cd.infer_circles(circles_images)
        players = []
        marks = []
        waypoints = []
        for i, label in enumerate(predicted_labels):
            if label == 'player':
                players.append(circles[i])
            elif label == 'mark':
                marks.append(circles[i])
            elif label == 'waypoint':
                waypoints.append(circles[i])
        return players, marks, waypoints
    
    def connect_player_mark(self, image, scale, scene='map_l2plus', relative_height=0):
        cal_dis = lambda x1,y1,x2,y2:math.sqrt(abs(x1-x2)**2+abs(y1-y2)**2)
        avg_color = lambda a,b:(int(a[0]*0.5+b[0]*0.5), int(a[1]*0.5+b[1]*0.5), int(a[2]*0.5+b[2]*0.5))

        height = image.shape[0]
        pix_per_100m = height/scale

        players, marks, waypoints = self.get_pos(image, scene=scene)
        if self.debugging:
            print('Players', players)
            print('Marks', marks)
            print('Waypoints', waypoints)
        
        rad_player = int(height*0.0105)
        rad_mark = int(height*0.009)

        rad_player = int(height*0.0105)
        rad_mark = int(height*0.009)
        
        lines = [] # [[x1, y1, x2, y2, color, distance],...]
        circles = [ [m[0], m[1], rad_mark, (255,255,153)] for m in marks ] # [[x, y, rad, color, thick]]
        
        colors = [
            (51,255,255), # Yellow 
            (51,255,51),  # Green
            (204,102,0), # Blue
            (102,0,204), # Wine
            (204,204,0), # Cyan
            (255,0,127)  # Purple 
        ]
        color_id = 0

        distincted_marks = []
        for mark in marks:
            m_x, m_y, _ = mark
            distance_pixs = []
            for player in players:
                p_x, p_y, _ = player
                distance_pix = cal_dis(p_x, p_y, m_x, m_y)
                distance_pixs.append(distance_pix)
            if min(distance_pixs) > 30:
                distincted_marks.append(mark)

        marks = distincted_marks
        
        for i, player in enumerate(players):
            p_x, p_y, _ = player
            color = colors[color_id]
            color_id = (color_id+1)%6
            circle = [p_x, p_y, rad_player, color]
            circles.append(circle)
            for mark in marks:
                m_x, m_y, m_rad = mark
                distance_pix = cal_dis(p_x, p_y, m_x, m_y)
                distance_m = int(distance_pix/pix_per_100m*100)
                if (distance_m > self.min_distance and distance_m < self.max_distance):
                    line = [p_x, p_y, m_x, m_y, color, distance_m]
                    lines.append(line)
            for player2 in players[i+1:]:
                p2_x, p2_y, _ = player2
                p2p_color = avg_color(color, (255,255,255))
                distance_pix = cal_dis(p_x, p_y, p2_x, p2_y)
                distance_m = int(distance_pix/pix_per_100m*100)
                if (distance_m > self.min_distance and distance_m < self.max_distance):
                    line = [p_x, p_y, p2_x, p2_y, p2p_color, distance_m]
                    lines.append(line)
                
        return lines, circles

    def draw_lines(self, image, lines):
        img = image.copy()
        for line in lines:
            start_point = (line[0], line[1])
            end_point = (line[2], line[3])
            color = line[4]
            distance = f'{line[5]}'
            #offset = 25
            #if (start_point[0] > end_point[0]):
            #    offset = -25
            cv2.line(img, start_point, end_point, color, 2)
            mid_point = (int(start_point[0]*0.5+end_point[0]*0.5), int(start_point[1]*0.5+end_point[1]*0.5))
            cv2.putText(img, distance, mid_point, 0, 1, color, 2)
        return img

    def draw_circles(self, image, circles):
        img = image.copy()
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            color = circle[3]
            img = cv2.circle(img, center, radius, color, 3)
        return img