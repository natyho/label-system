import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import Polygon

from transformer.Transformer import Transformer

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
if __name__ == '__main__':
#%% 手動場域校正介面展示
    ## 讀取影像   
    path_src = r'rtsp://keelungrd:keelungrdits@210.61.151.115/onvif-media/media.amp?streamprofile=Profile1&amp;audio=0'
    # path_src = r'D:\video109\surveillance\KH_B001\KH-B001-20200603-202338-event_F18891065\KH-B001-20200603-202338-event.mp4'
    # path_src = r'D:\video109\surveillance\KH_B001\KH-B001-20200603-202338-event_F18891065\20200819_111436.avi'
    # path_src = r'D:\video109\simulation\20200505\crosscrash_1.505355465122887\observer1.avi'
    cap = cv2.VideoCapture(path_src)
    ret, frame = cap.read()

    ## 建立場域校正物件
    transformer = Transformer()
    transformer.load_camera_parameter_by_path(os.path.join(os.getcwd(), 'transformer/camer'))
    
    ## 建立取點UI介面
    ma = ManualSetting()
    dist_size   = (12,10)
    undst = transformer.get_undistort_image(frame)
    ma.set_image(undst , dist_size)
    plt.show()
    
    image_anchor, world_anchor, image_scope, world_scope = ma.get_setting_anchor()

    ## 計算轉至點
    transformer.load_camera_parameter_by_path(os.path.join(os.getcwd(), 'transformer/camer'))
    transformer.set_image_points(image_anchor)
    transformer.set_world_points(world_anchor)
    transformer.set_image_scope(image_scope)
    transformer.set_world_scope(world_scope)
    transformer.calculate_image_to_world_matrix()
    
    ## 應用轉至點
    transformer.transepose_image_to_world(image_anchor)
    img_map = transformer.warp_image(undst)
    plt.figure(); plt.imshow(img_map[:,:,::-1])
    
    ## 儲存轉換模型
    transformer.save_parameter(path_save=os.path.join(os.getcwd(), r'C:\GitHub\109_EventEngine\event_engine_109\transformer\field\KL_B001.npz'))

    rt_load = Transformer()
    rt_load.load_parametre(path_load=os.path.join(os.getcwd(), r'C:\GitHub\109_EventEngine\event_engine_109\transformer\field\KL_B001.npz'))
    img_map_load = rt_load.warp_image(undst)
    self = rt_load
    plt.figure(); plt.imshow(img_map_load[:,:,::-1])
    
    plt.figure(); plt.imshow(undst[:,:,::-1])

    rt_load.transepose_world_to_image([[600,800]])
