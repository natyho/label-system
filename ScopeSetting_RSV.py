#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:13:35 2020

@author: ystseng
"""


import sys, os
# sys.path.append('event/')
import tkinter as tk
from time import time
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import Polygon
from time import sleep


import matplotlib.patches as patches

# from radar.RadarCapture import RadarCapture, IPcamCapture
# from transformer.Transformer import Transformer

#%%
class TKInputDialog:
    #ref: https://www.python-course.eu/tkinter_entry_widgets.php
    def __init__(self, labels:list=['x', 'y'], defaults:list=[], title=None, win_size=(200, 150)):
        self.master = tk.Tk()
        self.entries = {}
        self.ret = None
        if title:
            tk.Label(self.master, text=title).pack(side=tk.TOP)

        #create input labels and entries
        for i, label in enumerate(labels):
            row = tk.Frame(self.master)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            lab = tk.Label(row, width=5, text=label, anchor='w')
            lab.pack(side=tk.LEFT)
            ent = tk.Entry(row, width=3)
            ent.insert(0, defaults[i] if any(defaults) else '') #fill default value
            ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries[label] = ent

        #confirm actions
        tk.Button(self.master, text='Ok', command=self.ok).pack(side=tk.LEFT, padx=5, pady=5) #press OK button
        self.master.protocol('WM_DELETE_WINDOW', self.ok) #close window https://stackoverflow.com/a/111160/10373104
        self.master.bind('<Return>', self.ok) #press Enter https://stackoverflow.com/a/16996475/10373104

        #adjust window position & size https://stackoverflow.com/a/14910894/10373104
        w, h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry('%dx%d+%d+%d' % (win_size[0], win_size[1], int(w/2), int(h/2)))

        self.master.focus_force() #move focus to this window https://stackoverflow.com/a/22751955/10373104
        self.master.mainloop()

    def ok(self, event=None):
        def to_float(x):
            try:
                return float(x)
            except ValueError:
                return x
        self.master.quit()
        # self.ret = {key: float(val.get()) for key, val in self.entries.items()}
        self.ret = [to_float(val.get()) for key, val in self.entries.items()]
    
    def get(self):
        self.master.withdraw() #close Tk window
        return self.ret


class TKSrollDownDialog:
    #ref: https://stackoverflow.com/a/45442534/10373104
    def __init__(self, options:list=[], title=None, win_size=(150, 100)):
        self.master = tk.Tk()

        if title:
            tk.Label(self.master, text=title).pack(side=tk.TOP)

        # setup scroll down menu
        self.ret = tk.StringVar(self.master)
        self.ret.set(options[0]) # default value
        w = tk.OptionMenu(self.master, self.ret, *options).pack()

        #confirm actions
        tk.Button(self.master, text='Ok', command=self.ok).pack(side=tk.BOTTOM, padx=5, pady=5) #press OK button
        self.master.protocol('WM_DELETE_WINDOW', self.ok) #close window https://stackoverflow.com/a/111160/10373104
        self.master.bind('<Return>', self.ok) #press Enter https://stackoverflow.com/a/16996475/10373104

        #adjust window position & size https://stackoverflow.com/a/14910894/10373104
        w, h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry('%dx%d+%d+%d' % (win_size[0], win_size[1], int(w/2), int(h/2)))

        self.master.focus_force() #move focus to this window https://stackoverflow.com/a/22751955/10373104
        self.master.mainloop()

    def ok(self, event=None):
        def to_float(x):
            try:
                return float(x)
            except ValueError:
                return x
        self.master.quit()
        # self.ret = 
        # self.ret = {key: float(val.get()) for key, val in self.entries.items()}
        # self.ret = [to_float(val.get()) for key, val in self.entries.items()]
    
    def get(self):
        self.master.withdraw() #close Tk window
        return self.ret.get()


class MousePointsReactor:
    '''return desired mouse click points on frame get_mouse_points'''
    #https://stackoverflow.com/questions/33370613/using-mouse-to-plot-points-on-image-in-python-with-matplotlib
    def __init__(self, img, num, labels:list=None, defaults:list=None):
        self.pts = {} #final results
        self.circles = [] #double click positions
        self.texts = [] #show text for circles
        self.line_pts = [] #alignment lines [[(x1, y1), (x2, y2)], ...]
        self.newline = [] #for drawing a line
        self.l = None #temporary line patch
        self.img = img
        self.num = num
        self.ax = None #pyplot canvas
        self.labels = labels
        self.defaults = defaults
        self.str_defaults = [str(x) for x in defaults]
        self.lock = False #lock line drawing
        self.leave = False
        self.loop = True
        master = tk.Tk() #dummy tk root window
        master.withdraw() #remove blank window https://stackoverflow.com/a/17280890/10373104

    def onClick(self, event):
        if not event.xdata or not event.ydata: #click outside image
            return
        pos = (event.xdata, event.ydata)
        if event.button == MouseButton.LEFT:
            #check click-on-circle event
            click_on_circle = False
            for i_c, c in enumerate(self.circles):
                ## check every exist circule 
                contains, attrd = c.contains(event)
                if contains:
                    click_on_circle = True
                    #get values from popup
                    
                    list_unpicked_item = [iy for iy,y in enumerate(self.defaults) if str(y) not in [str(x) for x in self.pts.values()]]
                    list_picked_item = [iy for iy,y in enumerate(self.defaults) if str(y) in [str(x) for x in self.pts.values()]]
                    list_show = [self.str_defaults[x] for x in list_unpicked_item ] +['--------']+[self.str_defaults[x] for x in list_picked_item ]
                    list_show_result = list_unpicked_item+ [999]+list_picked_item 
                    idx = list_show.index(TKSrollDownDialog(list_show).get())
                    
                    if list_show_result [idx]!=999:
                        # if not choose '--------'
                        # if choose pickuped item, remove text 
                        self.pts.update({c.center:self.defaults[list_show_result [idx]]})
                        if self.str_defaults[list_show_result[idx]] in [x.get_text() for x in self.texts]:
                            drop_index = [x.get_text() for x in self.texts].index(self.str_defaults[list_show_result[idx]])
                            self.texts.pop(drop_index).remove()
                        # draw the new text
                        t = self.ax.text(c.center[0]+10, c.center[1]-10, self.str_defaults[list_show_result[idx]], color='yellow')
                        self.texts.append(t)
                        
                    else:
                        # if choose '--------', remove node
                        c=self.circles[0]
                        if (c.center[0]+10,c.center[1]-10) in [x.get_position() for x in self.texts]:
                            # remove text
                            drop_index =[x.get_position() for x in self.texts].index((c.center[0]+10,c.center[1]-10))
                            self.texts.pop(drop_index).remove()
                            # remove point
                            del(self.pts[c.center])
                    break                      

            #not click on circle -> draw lines
            if not click_on_circle and not self.lock:
                if not self.newline: #add line start point
                    # print('single1')
                    self.newline = [pos]
                else: #add line end point
                    # print('single2')
                    self.newline.append(pos)
                    self.line_pts.append(self.newline)
                    self.newline = []
                    self.l = None #reset patch (unbound to the finished line)
                    self.show_intersections()
            click_on_circle = False #reset
        elif event.button == MouseButton.RIGHT and not self.lock:
            # print('remove')
            def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
                a=lineY2-lineY1
                b=lineX1-lineX2
                c=lineX2*lineY1-lineX1*lineY2
                dis=(np.abs(a*pointX+b*pointY+c))/(np.sqrt(a*a+b*b))
                return dis                 
            try:
                list_dis = []
                for lines in self.line_pts:
                    list_dis.append(getDis(pos[0],pos[1], lines[0][0],lines[0][1],lines[1][0],lines[1][1]))
                index_drop = np.argmin(list_dis)
                
                del(self.line_pts[index_drop])
                self.ax.lines[index_drop].remove()
                self.show_intersections()
            except:
                pass

        self.ax.figure.canvas.draw_idle() #update canvas            

    def onMove(self, event):
        if not event.xdata or not event.ydata or self.lock: #click outside image
            return
        pos = (event.xdata, event.ydata)
        if self.newline: #has start point
            try:
                self.l.remove() #or self.ax.lines[0].remove() or self.ax.lines.remove(self.l)
            except:
                pass
            #Line2D https://stackoverflow.com/a/36488527/10373104
            self.l = Line2D([self.newline[0][0], pos[0]], [self.newline[0][1], pos[1]], color='red')
            self.ax.add_line(self.l)
        self.ax.figure.canvas.draw_idle() #update canvas                

    def onKyePress(self, event):
        self.lock = self.lock ^ (event.key in ['x', 'X']) #update lock status
        if self.lock:
            plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Set Mode')
        else:
            plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Line Mode')
        self.leave = event.key in ['l', 'L', 'Q','q']
    
    def start(self):
        self.loop = True
        self.leave = False
        fig, self.ax = plt.subplots(1)
        plt.imshow(self.img[:,:,::-1]) #or self.ax.imshow(self.img[:,:,::-1])
        # cid1 = fig.canvas.mpl_connect('button_press_event', self.onClick)
        # cid2 = fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        # cid3 = fig.canvas.mpl_connect('key_press_event', self.onKyePress)
        fig.canvas.mpl_connect('button_press_event', self.onClick)
        fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        fig.canvas.mpl_connect('key_press_event', self.onKyePress)
        plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Line Mode')
        plt.show()
        # auto close version
        plt.show(block=False) #https://github.com/matplotlib/matplotlib/issues/8560/#issuecomment-397254641
        while (len(self.pts) < self.num) & self.loop :
            if self.leave:
                self.loop = False
            if cv2.waitKey(1) in [ord('q'), ord('Q'), 27]:
                break
        sleep(1)
        plt.close(fig)
        print('lines = ', self.line_pts)
        return self.pts
    
    def get_pts(self):
        return self.pts
    def get_inline_intersections(self):
        '''get inline intersection points from drawed straight lines'''
        def is_within(x, p1, p2): #p1, p2=(x1, y1), (x2, y2)
            return (p1[0]<=x[0]<=p2[0] or p1[0]>=x[0]>=p2[0]) and (p1[1]<=x[1]<=p2[1] or p1[1]>=x[1]>=p2[1])
        intersections = []
        for i, p1 in enumerate(self.line_pts): #p1 = [(a, b), (c, d)]
            #formulate as px + qy = r
            m1 = (p1[0][1] - p1[1][1], p1[1][0] - p1[0][0]) #(p, q) = (b-d, c-a)
            b1 = p1[1][0]*p1[0][1] - p1[0][0]*p1[1][1] #cb-ad

            for j, p2 in enumerate(self.line_pts[i+1:]):
                #formulate as px + qy = r
                m2 = (p2[0][1] - p2[1][1], p2[1][0] - p2[0][0]) #(p, q) = (b-d, c-a)
                b2 = p2[1][0]*p2[0][1] - p2[0][0]*p2[1][1] #cb-ad

                #find intersection point
                #reference: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/87358/
                try:
                    r = np.linalg.solve(np.mat([m1, m2]), np.mat([b1, b2]).T)
                    if is_within(r, p1[0], p1[1]) and is_within(r, p2[0], p2[1]):
                        intersections.append((int(r[0][0]), int(r[1][0])))
                except np.linalg.LinAlgError: #no intersection
                    pass
        return intersections

    def show_intersections(self):
        '''show inlin intersection points on canvas and storing'''
        intersections = self.get_inline_intersections()        
        # clear original patches
        for i in range(len(self.circles)):
            self.circles.pop(-1).remove()
        for i in range(len(self.texts)):
            self.texts.pop(-1).remove()
        self.circles = []
        self.texts = []
        self.pts = {}
        # draw new circles
        for p in intersections:            
            c = patches.Circle(p, 7, color='blue', zorder=100) #draw a circle on top
            self.ax.add_patch(c)
            self.circles.append(c)


#%%
def set_SV_transformer(rt, src, radar=None, flag_reset=False):
    
    src=r'C:\Users\SSI\Desktop\GUI\0604cb\2020-06-04_16-46-28,448645.mp4'
    flag_reset = True
    ##是否為離線模式    
    offline = not src.startswith('rtsp')
    vout = src.startswith('rtsp')

    ## 連接攝影機
    # ipcam = IPcamCapture(src, offline=offline, start_frame=0)
    # ipcam.modify_time_drify(second=2, microsecond=200000)
    cap = cv2.VideoCapture(src)
    ret, img = cap.read()    
    
    # ## 啟動子執行緒
    # ipcam.start(write=vout)
    # ## 暫停1秒，確保影像已經填充
    # sleep(0.5)
    # img, _ = ipcam.get_frame()
    # ## ipcam.stop()
    
    ''' UI for world to video'''
    points = [[[10,0],  [570,954]],
              [[20,0],  [340,1010]],
              [[30,0],  [247,1030]],
              [[40,0],  [203,1043]],
              [[50,0],  [177,1050]],
              [[60,0],  [155,1052]],
              [[70,0],  [147,1054]],
              [[10,6],  [462,359]],
              [[20,6],  [282,659]],
              [[30,6],  [213,784]],
              [[40,6],  [177,855]],
              [[50,6],  [154,896]],
              [[60,6],  [140,923]],
              [[70,6],  [130,944]],
              [[10,10], [396,23]],
              [[20,10], [250,479]],
              [[30,10], [197,642]],
              [[40,10], [167,740]],
              [[50,10], [145,806]],
              [[60,10], [132,836]],
              [[70,10], [125,865]],]
    
    image_points = np.array(points)[:,1,:]
    world_points = np.array(points)[:,0,:]

    # mouse configuring or load existance
    path_image_npz = os.path.join(os.getcwd(),'transformer/points_image.npz')
    if os.path.exists(path_image_npz ) & ~flag_reset:
        print('read npy')
        points = np.load(path_image_npz )
        image_points, world_points = points['arr_0'], points['arr_1']
    else:
        print('redraw point')
        get_points = MousePointsReactor(img, len(world_points), ['x', 'y'], world_points)
        points = get_points.start()

        world_points = np.array(list(points.values()))
        image_points = np.array([np.array(x) for x in points.keys()])
        np.savez(path_image_npz , image_points, world_points)
    
    #set1
    '''
    rt.set_image_points(image_points)
    rt.set_world_points(world_points)
    rt.load_camera_parameter_by_path(os.path.join(os.getcwd(), 'transformer/camer'))#r'C:\\GitHub\\109_RadarFusion\\panasonic_camera\\')

    rt.optimal_camera_matrix()
    pixel_point = rt.transepose_world_to_image(world_points)
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.plot(pixel_point[:,0], pixel_point[:,1], 'ro' )
    plt.plot(image_points[:,0], image_points[:,1], 'bx' )
    
    rt.calculate_image_to_world_matrix()
    rt.save_parameter()
    return rt, ipcam

'''


def set_RS_transformer(rt, radar, flag_reset=False, flag_online_Radar=False):
    
    '''
    radar = RadarCapture(COM, filepath=filepath)
    '''
    
    meter_pixel = 12    
    radar_range_y = 30
    radar_range_x = 90
    radar_location = ((radar_range_x-5)*meter_pixel, (radar_range_y-15)*meter_pixel)
    def pixel_radar(x, y, theta=0, dxy=[0,0]):
        '''radar to world pixel transformer'''
        th=theta*np.pi/180
        M=np.array([[np.cos(th), - np.sin(th)],[np.sin(th), np.cos(th)]])
        v=np.array([[y],[x]])
        v_r = np.dot(M,v)    
        y= v_r[0]-dxy[0]
        x= v_r[1]+dxy[1]
        return (int(radar_location[1]-y*meter_pixel), int(radar_location[0]-x*meter_pixel))
    
    def draw_birdview_basemap(dict_target={}, dict_road={}):
        '''world pixel basemap'''
        ## plot radar background 
        radar_plane = np.zeros([radar_range_x*meter_pixel, radar_range_y*meter_pixel, 3], dtype = np.uint8)*10
    
        ## plot target region with 3 meter
        for distance in range(10):
            # distance=0
            cv2.line(radar_plane, pixel_radar(distance*10, -15), pixel_radar(distance*10, 15), (100, 100, 100), 1)        
        
        for target_key in dict_road.values():
            cv2.line(radar_plane, pixel_radar(target_key[0][0], target_key[0][1]), pixel_radar(target_key[1][0], target_key[1][1]), (150, 150, 150), 3)
        
        for target_key in dict_target.values():
            cv2.circle(radar_plane , pixel_radar(target_key[0], target_key[1]), int(3*meter_pixel), (0,0,255),2)        
            
        # cv2.imshow('radar plane', cv2.resize(radar_plane ,(0,0),fx=1,fy=1) )
        return radar_plane 
    
    class SaveRadarUIPoint():
        '''Radar Point sets'''
        def __init__(self, num_points, dict_target):
            self.points=[]
            self.num_points = num_points
            self.dict_target = dict_target
        def append(self,point):
            self.points.append(point)
            
        def start(self):
            points = TKInputDialog(
                labels=['x:%s\ny:%s'%(self.dict_target[x][0], self.dict_target[x][1]) for x in self.dict_target], 
                defaults=[], 
                title='Enter the radar sample data', 
                win_size=(240, 50 * self.num_points)
                ).get()
            if points != '':
                for i_p,point in enumerate(points):
                    if point != '':
                        self.points.append({ i_p:np.array([float(pp) for pp in point.lstrip().split(' ')]) })
                    
        def get_RW_points(self):
            radar_points = np.array([list(x.values())[0] for x in self.points])
            world_points = np.array([self.dict_target[list(x.keys())[0]] for x in self.points])
            
            return radar_points, world_points 
        
        
    '''set radar to world point sets'''
    path_radar_npz = os.path.join(os.getcwd(),'transformer/points_radar.npz')
    if not os.path.exists(path_radar_npz) & ~flag_reset:
        
        # world_points = np.array([[31,2],
        #                         [50,2],
        #                         [74,2],
        #                         [68,10],
        #                         [28,10],])
        world_points = np.array([[35,2],
                                [50,2],
                                [68,10],
                                [50,10],
                                [28,10],])
        radar_points = np.array([[40.5, -0.6],
                                 [63.3, -3.0],
                                 [64.6,  3.0],
                                 [48.0,  4.7],
                                 [27.4,  7.9],])
        num_points = TKInputDialog(
            labels=['N = '], 
            defaults=[5], 
            title='Set total number of target regions', 
            win_size=(200, 50)
        ).get()[0]
        try:
            num_points = int(num_points)
        except:
            num_points =5
            print('Enter Number is not a interger, set region to 5')
    
        if num_points <=5:
            default_pair = ['%d %d'%(str_default[0],str_default[1]) for str_default in world_points[:num_points]]
        else:
            default_pair = [list(x) for x in world_points]+['']*(num_points-5)
            
        str_target_centers = TKInputDialog(
            labels=['R %d'%num for num in range(num_points)], 
            defaults=default_pair , 
            title='Set target regions centers', 
            win_size=(240, 50*num_points)
        ).get()  
    
        dict_target = dict(enumerate([[float(point)for point in strpair.split(' ')]for strpair in str_target_centers if strpair !='']))
        
        dict_road ={ '1':[(0,0), (80,0)],
                     '2':[(0,6), (80,6)],
                     '3':[(0,12),(80,12)],
                     # '3':[(0,12),(80,12)]
            }
        save_radar_point = SaveRadarUIPoint(num_points, dict_target)
        radar_adjust_rotation = 0
        radar_drift_position = [0,0]
        last_radar_objects ={}
        list_save_object =[]
        list_slow_points = []
        list_fast_points = []    
        
        ## A. start radar
        if flag_online_Radar:
            radar_start_time = system_start_time = None
            radar.get_info()
            radar.start_parse()
    
            radar_start_time = None
            while not radar_start_time:
                radar_start_time = radar.start_worktime
            system_start_time = time()*1000
        else:
            from RSV_Capture import RadarData
            frame_i = 0
            radar = RadarData(filepath=r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cb\radar.log')
            radar_data = radar.list_object_frame
            radar_start_time = datetime.strptime(radar_data[0]['time'], "%Y-%m-%d %H:%M:%S.%f").timestamp()*1000##Debug
        system_start_time = time()*1000            
        difference_time_radar_system = system_start_time -radar_start_time 
        size_text = 0.5
        flag_loop = True
        flag_first = False
        
        
        radar_plane_raw = draw_birdview_basemap(dict_target, dict_road)
        while flag_loop:
            ## a. get radar objects 
            if flag_online_Radar:
                search_radar_time = time()*1000 - difference_time_radar_system 
                # print(search_radar_time)
                current_radar_objects = radar.get_current_object(search_radar_time , nearest=True)
            else:
                current_radar_objects = radar_data[frame_i ]##Debug
                frame_i +=1##Debug

            ## b. get radar plant background
            radar_plane = radar_plane_raw.copy()
            if current_radar_objects:

                '''
                last_radar_objects=radar_data[155]## Debug
                current_radar_objects=radar_data[156]## Debug
                '''
                if flag_first:
                    ## e. find dissappear object, and add to temp record
                    old_radar_IDs = {last_radar_objects[object_ID]['oid']:object_ID  for object_ID in last_radar_objects.keys() if object_ID not in ['worktime_ms', 'systime'] }
                    new_radar_IDs = {current_radar_objects[object_ID]['oid']:object_ID  for object_ID in current_radar_objects.keys() if object_ID not in ['worktime_ms', 'systime'] }
                    for old_radar_ID in old_radar_IDs:
                        if old_radar_ID not in new_radar_IDs:
                            list_save_object.append(last_radar_objects [old_radar_IDs[old_radar_ID]])
        
                    ## f.shwo slow objects     
                    for current_radar_key in [obj for obj in current_radar_objects.keys() if obj not in ['worktime_ms', 'systime']]:
                        # get position two points
                        radar_object = current_radar_objects[current_radar_key ]
                        if current_radar_key in last_radar_objects.keys():
                            radar_object_last = last_radar_objects [current_radar_key ] 
                        else:
                            continue
                        
                        ## check speed type
                        speed_up_limit = 35*000/60/60 ## km speed * radar fps
                        speed_low_limit = 7*1000/60/60*0.075 ## km speed * radar fps
                        if np.linalg.norm([(radar_object['x'] -radar_object_last['x']), (radar_object['y'] -radar_object_last['y'])]) < speed_low_limit :
                            list_slow_points.append((radar_object['x'],radar_object['y']))
                        elif np.linalg.norm([(radar_object['x'] -radar_object_last['x']), (radar_object['y'] -radar_object_last['y'])]) > speed_up_limit :
                            list_fast_points.append((radar_object['x'],radar_object['y']))
        
                else:            
                    flag_first = True
                last_radar_objects = current_radar_objects.copy()
        
                ## g.show slow point (purple)
                list_slow_points = list_slow_points[-60:]
                for pp in list_slow_points:
                    cv2.circle(radar_plane, pixel_radar(pp[0], pp[1],radar_adjust_rotation, radar_drift_position), int(3), (150,0,255),-1)
                    
                ## h.show fast point (Aqua blue)
                list_fast_points = list_fast_points[-60:]
                for pp in list_fast_points:
                    cv2.circle(radar_plane, pixel_radar(pp[0], pp[1],radar_adjust_rotation, radar_drift_position), int(1), (255,200,0),-1)
                
                ## c. print radar date time
                cv2.putText(radar_plane, str(datetime.fromtimestamp(current_radar_objects['systime'])), (10,20), cv2.FONT_HERSHEY_SIMPLEX, size_text, (0,0,255),2,cv2.LINE_4)

                ## d. draw radar objects
                for radar_key in [obj for obj in current_radar_objects.keys() if obj not in ['worktime_ms', 'systime']]:
                    # radar_key ='02'
                    radar_object = current_radar_objects[radar_key]
                    cv2.circle(radar_plane, pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation,radar_drift_position), int(2+radar_object['length']), (0,255,0),-1)
                    cv2.putText(radar_plane, '%s, %.02f, %.02f'%(radar_object['oid'], radar_object['x'], radar_object['y']), 
                                pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation,radar_drift_position ), 
                                cv2.FONT_HERSHEY_SIMPLEX, size_text, (0,0,255),1,cv2.LINE_AA)
                
           ## j.draw save nodes (yellow)
            for radar_object in list_save_object[-15:]:
                    # radar_key ='02'
                    # radar_object = saveframe[radar_key]
                    cv2.circle(radar_plane, pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation ), int(2+radar_object['length']), (0,255,255),-1)
                    cv2.putText(radar_plane, '%s, %.02f, %.02f'%(radar_object['oid'], radar_object['x'], radar_object['y']), pixel_radar(radar_object['x'], radar_object['y'],radar_adjust_rotation ), cv2.FONT_HERSHEY_SIMPLEX, size_text, (0,255,255),1,cv2.LINE_AA)
            ## k.draw radar position
            cv2.circle(radar_plane , pixel_radar(0,0,radar_adjust_rotation,radar_drift_position), int(10), (0,0,255),-1) 
            cv2.line(radar_plane, pixel_radar(0,0,radar_adjust_rotation,radar_drift_position), pixel_radar(80,0,radar_adjust_rotation,radar_drift_position), (0,200,255),4)
            cv2.imshow('radar plane', cv2.resize(radar_plane,(0,0) ,fx=1,fy=1))#600,1020
            
            ## i.keyboard activation
            keyevent = cv2.waitKey(1)
            if keyevent in [ord('a'), ord('A'), ord('D'),ord('d'), 27, ord('w'),ord('W'), ord('E'),ord('e'),
                        ord('i'),ord('k'),ord('j'),ord('l'),ord('I'),ord('K'),ord('J'),ord('L'),ord('s'),ord('S')]:
                if keyevent==27:
                    flag_loop=False
                    radar.stop()
                if keyevent in [ord('D'),ord('d')]:
                    list_save_object =[]
                    list_slow_points = []
                    list_fast_points = []                    
                if keyevent in [ord('A'),ord('a')] and current_radar_objects:
                    for radar_key in [obj for obj in current_radar_objects.keys() if obj not in ['worktime_ms', 'systime']]:
                        list_save_object.append(current_radar_objects[radar_key ])
                if keyevent in [ord('W'),ord('w')]:
                    radar_adjust_rotation -=0.1
                if keyevent in [ord('E'),ord('e')]:
                    radar_adjust_rotation +=0.1
                if keyevent in [ord('I'),ord('i')]:
                    radar_drift_position[1]=radar_drift_position[1]+0.1
                if keyevent in [ord('K'),ord('k')]:
                    radar_drift_position[1]=radar_drift_position[1]-0.1
                if keyevent in [ord('J'),ord('j')]:
                    radar_drift_position[0]=radar_drift_position[0]-0.1
                if keyevent in [ord('L'),ord('l')]:
                    radar_drift_position[0]=radar_drift_position[0]+0.1
                if keyevent in [ord('s'),ord('S')]:
                    save_radar_point.start()
        ## save reset radar points 
        print('redraw point')
        radar_points, world_points = save_radar_point.get_RW_points()
        np.savez(path_radar_npz, radar_points, world_points)
    else:      
        print('read npy')
        points = np.load(path_radar_npz )
        radar_points, world_points = points['arr_0'], points['arr_1']
            
    ## calculate transfer metrix
    rt.set_radar_points(radar_points)
    rt.set_world_points(world_points)
    rt.calculate_radar_world_matrix()
    
    ## show radar to world adjust result
    radar2world_points = rt.transepose_radar_to_world(radar_points)
    world2image_points = rt.transepose_world_to_radar(world_points)
    plt.figure()
    plt.title('World Radar adjust , \n x:radar domain, o:world domain, green circle: adjusted radar')
    plt.plot(-world_points[:,1],world_points[:,0],'ro',markersize=10, label='World Anchor')
    plt.plot(-radar_points[:,1],radar_points[:,0],'bx',markersize=10, label='Radar Anchor')
    plt.plot(-radar2world_points[:,1],radar2world_points[:,0],'go',markersize=5, label ='Adjusted radar')
    plt.legend()
    # plt.plot(-world2image_points[:,1],world2image_points[:,0],'cx',markersize=5 )
    plt.axis([-30,30,20,80])

    return rt



#%%
def dist(x, y):
    """
    Return the distance between two points.
    """
    d = x - y
    return np.sqrt(np.dot(d, d))


def dist_point_to_segment(p, s0, s1):
    """
    Get the distance of a point to a segment.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from
    http://geomalgorithms.com/a02-_lines.html
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return dist(p, s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return dist(p, s1)
    b = c1 / c2
    pb = s0 + b * v
    return dist(p, pb)

class PolygonInteractor(object):
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y,
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        # self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
        if self.line.stale:
            self.canvas.draw_idle()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        # self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


class ManualSetting():
    def __init__(self, unitpixel=100, skiplimit=99999, matrix=[], ):
        #print(pts2)
        self.SkipLimit          = skiplimit
        self.PixelPerUnit       = unitpixel
        self.flagMatrixExist    = False
        if matrix:
            self.matrix = matrix
            self.flagMatrixExist    = True
        else:
            self.matrix = []
            
    def set_image(self, frame, dist_size):
        self.frame = frame
        self.dist_size = dist_size
        self.hight = frame.shape[0]
        self.width = frame.shape[1]
        init_difP = 10    
        scope_node1 = np.array([[init_difP,init_difP],                                  [self.frame.shape[1]-init_difP,init_difP], 
                                [self.frame.shape[1]-init_difP,self.frame.shape[0]-init_difP],    [init_difP,self.frame.shape[0]-init_difP]])
        poly1       =[]
        poly1       = Polygon(scope_node1 , animated=True)
        
        frame2      = cv2.resize(self.frame, (self.dist_size[0]*self.PixelPerUnit, self.dist_size[1]*self.PixelPerUnit), interpolation=cv2.INTER_CUBIC)
        scope_node2 = np.array([[init_difP,init_difP],                                  [frame2.shape[1]-init_difP,init_difP], 
                                [frame2.shape[1]-init_difP,frame2.shape[0]-init_difP],  [init_difP,frame2.shape[0]-init_difP]])
        poly2       =[]
        poly2       = Polygon(scope_node2 , animated=True)
        
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.imshow(self.frame[:,:,::-1])
        self.ax1.add_patch(poly1)
        self.polygon1 = PolygonInteractor(self.ax1, poly1 )
        for ii in range(len(scope_node1)):
            self.ax1.text(scope_node1[ii,0], scope_node1[ii,1], ii,fontsize=16)
        self.ax1.set_title('Drag base Area')

        self.fig2, self.ax2 = plt.subplots()
        self.ax2.imshow(frame2[:,:,::-1])
        self.ax2.add_patch(poly2)
        self.polygon2 = PolygonInteractor(self.ax2, poly2 )
        for ii in range(len(scope_node2)):
            self.ax2.text(scope_node2[ii][0], scope_node2[ii][1], ii,fontsize=16)
        self.ax2.set_title('Set destination position')
        self.ax2.set_xticks([x*self.PixelPerUnit for x in range(self.dist_size[0]) ], minor=False)
        self.ax2.set_yticks([x*self.PixelPerUnit for x in range(self.dist_size[1]+1) ], minor=False)
        self.ax2.grid(color='r', linestyle='--', linewidth=1)
        self.fig1.canvas.mpl_connect('key_press_event', self.onKyePress)
        self.fig2.canvas.mpl_connect('key_press_event', self.onKyePress)
        self.fig1.show()
        self.fig2.show()
        
    def onKyePress(self, event):
        if event.key in ['x', 'X']:
            self.UpdateAffine()

    def UpdateAffine(self, ):
        pts1 = [list(x) for x in self.polygon1.poly.xy[:-1]]
        pts2 = [list(x) for x in self.polygon2.poly.xy[:-1]]
        self.inputPts = np.array(pts1, dtype = 'float32')
        self.outputPts = np.array(pts2, dtype = 'float32')
        self.matrix = cv2.getPerspectiveTransform(self.inputPts, self.outputPts)
        self.inv_matrix = cv2.getPerspectiveTransform(self.outputPts, self.inputPts)
        perspective = self.makeaffineimg(self.frame, False)
        self.ax2.imshow(perspective[:,:,::-1])
        self.ax2.figure.canvas.draw()
    
    def S2D(self, pt, flag_D_in_pixle=True):
        inputPoint = np.float32([[pt[0]], [pt[1]], [1]])
        resPoint = np.dot(self.matrix, inputPoint)
        resPoint/=resPoint[2][0]
        if flag_D_in_pixle:
            outputPoint = np.float64([resPoint[0][0], resPoint[1][0]])
        else:
            outputPoint = np.float64([resPoint[0][0]/self.PixelPerUnit, resPoint[1][0]/self.PixelPerUnit])

        return outputPoint

    def D2S(self, pt, flag_showresult=False):
        # pt=[100,100]
        # gold =[445, 238]
        
        inputPoint = np.float32([[pt[0]], [pt[1]], [1]])
        resPoint = np.dot(self.inv_matrix, inputPoint)
        resPoint/=resPoint[2][0]
        
        outputPoint = np.float64([resPoint[0][0], resPoint[1][0]])

        if flag_showresult:
            print(outputPoint)

        return outputPoint
          
    def makeaffineimg(self, frame, flag_verification=False):
        frame=cv2.resize(frame,(self.width,self.hight), interpolation=cv2.INTER_CUBIC)
        perspective = cv2.warpPerspective(frame, self.matrix, (self.dist_size[0]*self.PixelPerUnit, self.dist_size[1]*self.PixelPerUnit), cv2.INTER_LINEAR)
        return perspective 

    def get_setting_anchor(self):
        pts1 = [list(x) for x in self.polygon1.poly.xy[:-1]]
        pts2 = [list(x) for x in self.polygon2.poly.xy[:-1]]
        self.inputPts = np.array(pts1, dtype = 'float32')
        self.outputPts = np.array(pts2, dtype = 'float32')
        return self.inputPts, self.outputPts, (self.hight,self.width), self.dist_size
    
    def save_anchor(self, path_save='MA_metrix.npz'):
        np.savez(self.inputPts, self.outputPts, self.hight, self.width, self.dist_size)
        
    def load_anchor(self, path_load='MA_metrix.npz'):
        mtxs = np.load(path_load)
        self.inputPts = mtxs['arr_2']
        self.outputPts = mtxs['arr_2'] 
        self.hight = mtxs['arr_2']
        self.width = mtxs['arr_2']
        self.dist_size = mtxs['arr_2']
        
def ccorder(pts2):
    if len(pts2)==2:
        A=pts2
    else:
        A=[[],[]]
        for x in pts2:
            A[0].append(x[0])
            A[1].append(x[1])    
    A= A- np.mean(A, 1)[:, None]
    ord_A = np.argsort(np.arctan2(A[1, :], A[0, :]))
    return ord_A




#%%    
# def main(src, COM='COM13', filepath=None):
#     '''用來設置雷達錨點的主程式

#     Parameters
#     ----------
#     src : string
#         串流路徑或者是影片路徑
#     COM : string, optional
#         雷達的comport. The default is 'COM13'.
#     filepath : string, optional
#         log的存放位置. The default is None.

#     Returns
#     -------
#     None.

#     src=r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cb\2020-06-04_16-46-28,448645.mp4'
#     COM=None
#     filepath= r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cb\radar.log'
#     '''

#     ## 建立投影矩陣
#     # rt_set = Transformer()
#     # rt_set.load_camera_parameter_by_path(os.path.join(os.getcwd(), 'transformer/camer'))

#     ## A. 設置世界投影影像矩陣
#     rt_set, ipcam = set_SV_transformer(rt_set, src, flag_reset=False)
#     ipcam.stop()

#     ## B.設置雷達頭影視界矩陣
#     filepath = None if src.startswith('rtsp') else filepath
#     radar  = RadarCapture(COM, filepath=filepath)
#     rt_set = set_RS_transformer(rt_set, radar, flag_reset=False, flag_online_Radar=True)

#     ## C.儲存投影矩陣結果
#     path_radar_transformer = os.path.join(os.getcwd(), r'transformer\field\radar.npz')
#     rt_set.save_parameter(path_save=path_radar_transformer )
    
    
#     #### 測試結果
#     ## A.讀取轉至矩陣
#     rt = Transformer()
#     rt.load_parametre(path_load=path_radar_transformer )
#     rt.mtx_W2I
#     rt.mtx_I2W
    
#     ## B. 連接攝影機
#     _, img = cv2.VideoCapture(src).read()
#     plt.figure();plt.imshow(img[:,:,::-1])
#     world_paths = np.array([[np.linspace(10,80,200), [0]*200],
#                             [np.linspace(10,80,200),[6]*200],
#                             [np.linspace(10,80,200),[10]*200,]])  
    
#     for x in range(200):
#         pp = rt.transepose_world_to_image([world_paths[0,:,x]])[0]
#         cv2.circle(img , (int(pp[0]), int(pp[1])), int(2), (0,0,255),-1)
        
#         pp = rt.transepose_world_to_image([world_paths[1,:,x]])[0]
#         cv2.circle(img , (int(pp[0]), int(pp[1])), int(2), (0,255,0),-1)
        
#         pp = rt.transepose_world_to_image([world_paths[2,:,x]])[0]
#         cv2.circle(img , (int(pp[0]), int(pp[1])), int(2), (255,0,0),-1)
        
#     plt.imshow(img[:,:,::-1])
#     plt.show()    
    
# #%%    
# if __name__ == '__main__':
#         dict_carmer = {
#             'KL' : "rtsp://keelung:keelung@nmsi@60.251.176.43/onvif-media/media.amp?streamprofile=Profile1",
#             'KH' : "rtsp://demo:demoIII@221.120.43.26:554/stream0",
#             'IP' : 'rtsp://192.168.0.100/media.amp?streamprofile=Profile1',
#             'LL' : r'F:\GDrive\III_backup\[Work]202006\0604雷達資料\0604cc\2020-06-04_17-03-11,698371.mp4',
#             'LLt' : r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cc\2020-06-04_17-03-11,698371.mp4'
#         }
#         filepath = r'C:\GitHub\109_RadarFusion\Dataset\雷達資料0604\0604cc\radar.log'
#         main(dict_carmer['IP'], COM='COM13', filepath=filepath)
#         sys.exit()
