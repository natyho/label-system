# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:15:24 2020

@author: SSI
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
    def __init__(self, img, num, labels:list=None, defaults:list=None):
        self.img = img
        self.num = num
        self.labels = labels
        self.defaults = defaults
        self.pts = {} #final results
        self.circles = [] #double click positions
        self.texts = [] #show text for circles
        self.l = None #temporary line patch
        self.ax = None #pyplot canvas
        self.clicked_circle = None
        self.clicked_point = None
        self.c = None
        self.point = None
        self.kayma = None
        self.press = False
        self._start = False
        # self.line_pts = [] #alignment lines [[(x1, y1), (x2, y2)], ...]
        self.newline = [] #for drawing a line
        self.circles = []
        self.polygons = []
        self.move_patch = []
        self.current_items = {"circle": [], "line": [], "polygon": []}
        self.all_patches = []
        
        
        
    def onClick(self, event):
        if not event.xdata or not event.ydata: #click outside image
            return
        pos = (event.xdata, event.ydata) #set mouse data to pos
        if event.button == MouseButton.LEFT: #click left button:
            click_on_circle = False
            for i_c, c in enumerate(self.circles): #check if in every exist circule
                contains, attrd = c.contains(event)
                if contains:
                    click_on_circle = True
                    if self.current_items["circle"] and not self.current_items["polygon"]: #check if already drawing lines and no polygon 
                        if self.current_items["circle"][0].contains(event): 
                        #if so then, check if click back to the last point => if yes then draw polygon
                            vertex = []
                            for c in self.current_items["circle"]:
                                vertex.append(c.center)
                            p = Polygon(vertex)
                            self.ax.add_patch(p)
                            self.current_items["polygon"].append(p)
                            self.polygons.append(p)
                            self.current_items["line"].append(self.l)
                            # self.ax.add_line(self.l)
                            self.newline = []
                            self.l = None
                            self.all_patches.append(self.current_items)
                            for i, c_ in enumerate(self.current_items["circle"]):
                                if i>0:
                                    self.ax.patches.remove(c_)
                            self.current_items = {"circle": [], "line": [], "polygon": []}

                    elif not self.current_items["circle"]: #check if to move the polygon:
                        for patch in self.all_patches:
                            if c == patch["circle"][0]: 
                                self.press = True
                                self.clicked_circle = c
                                self.clicked_point = pos
                                self.move_patch.append(patch)

            if not click_on_circle:
                    c = patches.Circle(pos, 7, color='blue', zorder=100) #draw a circle on top
                    self.circles.append(c)
                    self.current_items["circle"].append(c)
                    self.ax.add_patch(c) 
                    if not self.newline: #add line start point
                    # print('single1')
                        self.newline = [pos]
                    else:
                        self.current_items["line"].append(self.l)
                        self.newline = [pos]
                        self.l = None
            # self.show_intersections()
            click_on_circle = False
        elif event.button == MouseButton.RIGHT:
            #check click-on-circle event
            remove_index = []
            def dist(x, y):
                """
                Return the distance between two points.
                """
                dx = x[0] - y[0]
                dy = x[1] - y[1]
                ans = dx**2 + dy**2
                ans = ans**(0.5)
    
                return ans
            for ip, patch in enumerate(self.all_patches):
                if dist(patch["circle"][0].center , pos) <= patch["circle"][0].radius:
                    remove_index.append(ip)
                    c = patch["circle"][0]
                    self.ax.patches.remove(c)
                    self.circles.remove(c)
                    for poly in patch["polygon"]:
                        self.ax.patches.remove(poly)
                        self.polygons.remove(poly)
                    for line in patch["line"]:
                        self.ax.lines.remove(line)
            for i in remove_index:
                self.all_patches.pop(i)
                        
            
                
                    
                    
                    
            for i_c, c in enumerate(self.circles):
                ## check every exist circule 
                contains, attrd = c.contains(event)
                if contains:
                    click_on_circle = True
                    self.ax.patches.remove(c)
                    self.circles.remove(c)
                    for item in self.total_item:
                        if c in item["c"]:
                            for p in item["p"]:
                                self.polygon.remove(p)
                                self.ax.patches.remove(p)
                            self.total_item.remove(item)
                    
                    
    def onMove(self, event):
        if not event.xdata or not event.ydata: #click outside image
            return
        pos = (event.xdata, event.ydata)
        def dist(x, y):
            """
            Return the distance between two points.
            """
            dx = x[0] - y[0]
            dy = x[1] - y[1]
            ans = dx**2 + dy**2
            ans = ans**(0.5)
    
            return ans
        
        if self.press:
            self._start = True 
             ## click point (move)
            self.kayma = (pos[0] - self.clicked_point[0] , pos[1] - self.clicked_point[1])
            self.clicked_point = pos[0], pos[1]
            for i, patch in enumerate(self.move_patch):
                for c in patch["circle"]:
                    c.set_center((c.center[0]  + self.kayma[0], c.center[1] + self.kayma[1]))
                    self.ax.figure.canvas.draw_idle()
            # for i_patch, patch in enumerate(self.moved_pat):
            #     # print(patch["circle"])   
            #     for i_c, c in enumerate(patch["circle"]):
                '''
                i_c = 0
                c = patch["circle"][0]
                self.kayma = (50,100)
                '''
                    # c.get_center()
                    # i = self.ax.patches.index(c)
                    # self.ax.patches[i].set_center(((self.ax.patches[i].get_center()[0] + self.kayma[0]), (self.ax.patches[i].get_center()[1] + self.kayma[1])))
                    # self.move_patch["circle"][0].center = \
                    #     (self.all_patches[i_all_p]["circle"][i_c].center[0] + self.kayma[0], self.all_patches[i_all_p]["circle"][i_c].center[1] + self.kayma[1])
                                    
                for i_p, p in enumerate(patch["polygon"]):
                    new_ver = []
                    for vertex in p.get_xy():
                        new_position = vertex[0] + self.kayma[0], vertex[1] + self.kayma[1]
                        new_ver.append(new_position)
                    i = self.ax.patches.index(p)
                    self.ax.patches[i].set_xy(new_ver)
                    # self.all_patches[i_all_p]["polygon"][i_p].set_xy(new_ver)
                    self.ax.figure.canvas.draw_idle()
                for i_l, l in enumerate(patch["line"]):
                    new_xdata = [l.get_xdata()[0] + self.kayma[0], l.get_xdata()[1] + self.kayma[0]]
                    new_ydata = [l.get_ydata()[0] + self.kayma[1], l.get_ydata()[1] + self.kayma[1]]
                    l.set_xdata(new_xdata)
                    l.set_ydata(new_ydata)
                    i = self.ax.lines.index(l)
                    self.ax.lines[i].set_xdata(new_xdata)
                    self.ax.lines[i].set_ydata(new_ydata)
                    self.ax.figure.canvas.draw_idle()
        else:
            in_circle = False
           
            for i_c, c in enumerate(self.circles):
                try:
                    distance = dist(pos, c.center)
                    
                    if distance <= c.get_radius():
                        in_circle = True
                        try:
                            self.ax.patches.remove(self.c)
                            self.c = None
                        except:
                            pass
                        self.c = patches.Circle(c.center, 7, color='red', zorder=100)
                        self.ax.add_patch(self.c)
                except:
                    pass
                
            if not in_circle:
                    try:
                        self.ax.patches.remove(self.c)
                        self.c = None
                    except:
                        pass
            in_circle = False    
            
        

            
            if self.newline: #has start point
                try:
                    self.ax.lines.remove(self.l)
                except:
                    pass
                #Line2D https://stackoverflow.com/a/36488527/10373104
                self.l = Line2D([self.newline[0][0], pos[0]], [self.newline[0][1], pos[1]], color='red')
                self.ax.add_line(self.l)
            
        self.ax.figure.canvas.draw_idle() #update canvas   
        
        
    def release(self, event):
        if self.press and self._start: 
            self.press = False
            self._start = False
            self.move_patch = []
            self.clicked_point = None
        
    # def get_inline_intersections(self):
    #     '''get inline intersection points from drawed straight lines'''
    #     def is_within(x, p1, p2): #p1, p2=(x1, y1), (x2, y2)
    #         return (p1[0]<=x[0]<=p2[0] or p1[0]>=x[0]>=p2[0]) and (p1[1]<=x[1]<=p2[1] or p1[1]>=x[1]>=p2[1])
    #     intersections = []
    #     for i, p1 in enumerate(self.line_pts): #p1 = [(a, b), (c, d)]
    #         #formulate as px + qy = r
    #         m1 = (p1[0][1] - p1[1][1], p1[1][0] - p1[0][0]) #(p, q) = (b-d, c-a)
    #         b1 = p1[1][0]*p1[0][1] - p1[0][0]*p1[1][1] #cb-ad

    #         for j, p2 in enumerate(self.line_pts[i+1:]):
    #             #formulate as px + qy = r
    #             m2 = (p2[0][1] - p2[1][1], p2[1][0] - p2[0][0]) #(p, q) = (b-d, c-a)
    #             b2 = p2[1][0]*p2[0][1] - p2[0][0]*p2[1][1] #cb-ad

    #             #find intersection point
    #             #reference: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/87358/
    #             try:
    #                 r = np.linalg.solve(np.mat([m1, m2]), np.mat([b1, b2]).T)
    #                 if is_within(r, p1[0], p1[1]) and is_within(r, p2[0], p2[1]):
    #                     intersections.append((int(r[0][0]), int(r[1][0])))
    #             except np.linalg.LinAlgError: #no intersection
    #                 pass
    #     return intersections
    
    
    # def show_intersections(self):
    #     '''show inlin intersection points on canvas and storing'''
    #     intersections = self.get_inline_intersections()        
    #     # clear original patches
    #     # for i in range(len(self.circles)):
    #     #     self.circles.pop(-1).remove()
    #     # for i in range(len(self.texts)):
    #     #     self.texts.pop(-1).remove()
    #     # self.circles = []
    #     # self.texts = []
    #     # self.pts = {}
    #     # draw new circles
    #     self.intersection = intersections
    #     for p in intersections:            
    #         c = patches.Circle(p, 7, color='blue', zorder=100) #draw a circle on top
    #         self.ax.add_patch(c)
    #         self.circles.append(c)
            
    def onKyePress(self, event):
        # self.lock = self.lock ^ (event.key in ['x', 'X']) #update lock status
        # if self.lock:
        #     plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Set Mode')
        # else:
        #     plt.title('Line mode: Left to draw, Right to remove.\nSet mode: cleck to select nodes.\npress x to switch modes.\npress L or Q to leave\nCurrent at Line Mode')
        self.leave = event.key in ['l', 'L', 'Q','q']
        if event.key in ['y', 'Y']:
            self.newline = []
            self.l.remove()
            self.all_patches.append(self.current_items)
            self.current_items = {"circle": [], "line": [], "polygon": []}
        self.ax.figure.canvas.draw_idle()
        
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
        fig.canvas.mpl_connect('button_release_event', self.release)

        
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
                    
                            
